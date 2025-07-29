import os
import argparse
import datetime

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL

# IP-Adapter
from utils.unet.ip_adapter.ip_adapter import ImageProjModel
from utils.unet.ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from utils.unet.ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from utils.unet.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

# Utility functions
from utils.utils import (
    get_loss_function, compute_vae_encodings, face_detection_mask, get_filelist,
    merge_image, save_png, scale_tensor, create_line_mask, apply_gaussian, AttentionStore
)

# Attack modules
from utils.unet.unet_attack import AttackUnet_IP_all, AttackCLIP
from utils.landmark.mtcnn_attack import mtcnn_attack
from utils.landmark.arcface_attack import AttackArcFace

# DCT tools
from utils.dct import dct_pass_filter, make_dct_basis, blockfy, encode, decode, deblockfy


def get_parser():
    parser = argparse.ArgumentParser()
    
    # Seed & model paths
    parser.add_argument("--seed", type=int, default=1, help="Random seed (used with seed_everything)")
    parser.add_argument("--model_path", type=str, default=None, help="Path or name of pretrained Stable Diffusion model")
    parser.add_argument("--vae_model_path", type=str, default=None, help="Path or name of pretrained VAE model")
    parser.add_argument("--unet_config", type=str, default=None, help="YAML config for attack UNet")
    parser.add_argument("--pretrained_ip_adapter_path", type=str, default=None, help="Path to pretrained IP-Adapter weights")
    parser.add_argument("--image_encoder_path", type=str, default=None, help="Path to pretrained image encoder (e.g., CLIP)")
    parser.add_argument("--pretrained_arcface50_path", type=str, default=None, help="Path to pretrained ArcFace-50 model")
    parser.add_argument("--pretrained_arcface100_path", type=str, default=None, help="Path to pretrained ArcFace-100 model")
    
    # FaceShield settings
    parser.add_argument("--save_path", type=str, default=None, help="Directory to save results")
    parser.add_argument("--resize_shape", type=int, default=512, help="Resize image to this square resolution")
    parser.add_argument("--proj_func", type=str, default="l1", help="Loss function for projection loss")
    parser.add_argument("--attn_func", type=str, default="l2", help="Loss function for attention loss")
    parser.add_argument("--attn_threshold", type=float, default=0.2, help="Threshold for masking low-attention regions")
    parser.add_argument("--mtcnn_func", type=str, default=False, help="Loss function for MTCNN feature loss")
    parser.add_argument("--arc_func", type=str, default="cosine", help="Loss function for ArcFace feature loss")
    parser.add_argument("--total_iter", type=int, default=30, help="Number of PGD iterations")
    parser.add_argument("--noise_clamp", type=int, default=12, help="Clamp value for adversarial noise (L∞ norm)")
    parser.add_argument("--step_size", type=float, default=1., help="Step size per PGD iteration")
    parser.add_argument("--image_path", type=str, default=None, help="Input image or directory path")

    return parser

def attack(args, gpu_num, gpu_no, **kwargs):
    config = OmegaConf.load(args.unet_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
    unet = AttackUnet_IP_all.from_pretrained(args.model_path, subfolder="unet", config_file=config, strict=False)
    image_preprocess = AttackCLIP()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path, subfolder="models/image_encoder")
    face_embedder50 = torch.load(args.pretrained_arcface50_path, weights_only=False)
    face_embedder100 = torch.load(args.pretrained_arcface100_path, weights_only=False)
    id_preprocess = AttackArcFace()

    # Freeze parameters of models to save more memory
    vae.requires_grad_(False).to(device)
    text_encoder.requires_grad_(False).to(device)
    image_encoder.requires_grad_(False).to(device)
    face_embedder50.requires_grad_(False).to(device)
    face_embedder100.requires_grad_(False).to(device)
    
    # ================================================== IP_Adapter =============================================== #
    # IP-Adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    ).to(device)
    
    # Init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        elif name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
            
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    # Calculate original checksums
    orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in image_proj_model.parameters()]))
    orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in adapter_modules.parameters()]))
    
    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location=device, weights_only=True)
    
    # Load state dict for image_proj_model and adapter_modules
    image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
    
    # Calculate new checksums
    new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in image_proj_model.parameters()]))
    new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in adapter_modules.parameters()]))

    # Verify if the weights have changed
    assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
    assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"
    
    unet.requires_grad_(False).to(device)
    # ============================================================================================================== #
    # Text token
    inputs = tokenizer([""], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    encoder_hidden_states = text_encoder(inputs.input_ids.to(device))[0]
    
    # Source Image and Mask load
    dataset_name = args.image_path.split('/')[-2]
    src_list = get_filelist(args.image_path, ['png','jpg'])
    if len(src_list) == 0:
        src_list = [args.image_path]
    num_samples = len(src_list)
    
    # Multi-gpu setting
    samples_split = num_samples // gpu_num
    remainder = num_samples % gpu_num
    if gpu_no < remainder:
        start_idx = gpu_no * (samples_split + 1)
        end_idx = start_idx + samples_split + 1
    else:
        start_idx = gpu_no * samples_split + remainder
        end_idx = start_idx + samples_split
    
    indices = list(range(start_idx, end_idx))
    gpu_samples = len(indices)
    src_list_rank = [src_list[i] for i in indices]
    filename_list = [f"{os.path.split(src_list_rank[id])[-1][:-4]}" for id in range(gpu_samples)]

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for indice in range(gpu_samples):
            batchT = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            save_dir = f"{args.save_path}/[{filename_list[indice]}]"
            os.makedirs(save_dir, exist_ok=True)
            
            # Image load
            face, _, _, back, coor = face_detection_mask(src_list_rank[indice], save_dir, args.resize_shape, device)
            gt_face = face.clone().detach()

            # Choice Loss function
            proj_func = get_loss_function(args.proj_func)
            attn_func = get_loss_function(args.attn_func)
            mtcnn_func = get_loss_function(args.mtcnn_func)
            arc_func = get_loss_function(args.arc_func)
            latents_query = compute_vae_encodings(face, vae, device, gt=True)
            
            # ============ Convert rgb to DCT ============ #
            N=8
            DCT_basis = make_dct_basis(N, device)
            low_pass_filter, high_pass_filter = dct_pass_filter(device)
            timestep = torch.tensor([0], device=device)
            
            # =============== Make ArcFace GT =============== #
            gt_50, gt_100 = id_preprocess.preprocess(gt_face)
            gt_id_50 = face_embedder50(gt_50.to(device))
            gt_id_100 = face_embedder50(gt_100.to(device))
            
            # ============ Make UNet GT ============ #
            var_controller = AttentionStore()
            gt_preprocessed = image_preprocess(gt_face)
            gt_encoded = image_encoder(gt_preprocessed).image_embeds # 1, 1024
            gt_proj = image_proj_model(gt_encoded) # 1, 4, 768
            stacked_encoder_hidden_states = torch.cat([encoder_hidden_states, gt_proj], dim=1) # 1, 81, 768
            unet(latents_query, timestep, stacked_encoder_hidden_states, store_controller=var_controller, unet_threshold=args.attn_threshold)
            
            with torch.enable_grad():
                delta = torch.zeros_like(gt_face, requires_grad=True).to(device)
                epochs = tqdm(range(args.total_iter), position=(indice*gpu_num+gpu_no), desc=f"[rank:{gpu_no}] batch {batchT}: {indice+1}/{gpu_samples}", total=len(range(args.total_iter)))
                for i, _ in enumerate(epochs):
                    adv_face = (255*gt_face) + delta
                    adv_face = torch.clamp(adv_face, min=0, max=255)
                                       
                    # ================== MTCNN attack ================== #
                    mtcnn_loss = 0
                    mtcnn_loss = mtcnn_attack(2 * (adv_face/255) - 1, loss_fn=mtcnn_func, loss=mtcnn_loss, device=device)

                    # ========== ArcFace Identity Attack ========== #
                    id_loss_50 = 0
                    id_loss_100 = 0
                    adv_50, adv_100 = id_preprocess.preprocess(adv_face/255)
                    adv_id_50 = face_embedder50(adv_50)
                    adv_id_100 = face_embedder100(adv_100)
                    id_loss_50 = arc_func(adv_id_50, gt_id_50)
                    id_loss_100 = arc_func(adv_id_100, gt_id_100)
                    id_loss = (-1) * id_loss_50 + (-1) * id_loss_100
                    
                    # ========== Diff-Conditioned UNet Attack ========== #
                    clip_loss = 0
                    attn_loss = 0
                    adv_preprocessed = image_preprocess(adv_face/255)
                    adv_encoded = image_encoder(adv_preprocessed).image_embeds # 1, 1024
                    adv_proj = image_proj_model(adv_encoded) # 1, 4, 768
                    stacked_encoder_hidden_states = torch.cat([encoder_hidden_states, adv_proj], dim=1) # 1, 81, 768
                    
                    clip_loss = proj_func(adv_encoded, gt_encoded) # Clip Loss
                    attn_loss = unet(latents_query, timestep, stacked_encoder_hidden_states, loss_fn=attn_func, loss=attn_loss, gt_attn_map=var_controller.attn_map.copy()) # UNet Loss
                    unet_loss = (-1) * clip_loss + (+1) * attn_loss
                    
                    # ========== PGD Update ========== #
                    total_loss = 9*mtcnn_loss + 4*id_loss + 1*unet_loss
                    total_loss.backward(retain_graph=True)
                    new_delta = args.step_size * torch.sign(delta.grad)

                    # ========== Smooth with Gaussian Blur ========== #
                    d_rgb = scale_tensor(new_delta)
                    mask = create_line_mask(save_dir, d_rgb)
                    new_delta = apply_gaussian(save_dir, new_delta, mask, 9, 5)
                    
                    # ========== Low-Pass Filter in DCT Domain ========== #
                    delta.data -= new_delta
                    grad_block, pad_size = blockfy(delta.data, N)
                    grad_dct = encode(grad_block, DCT_basis)
                    grad_dct_passed = grad_dct * low_pass_filter.expand(grad_dct.shape)
                    grad_block_passed = decode(grad_dct_passed, DCT_basis)
                    delta.data = deblockfy(grad_block_passed, pad_size)
                    
                    # ========== Clamp in ℓ∞ Norm Ball ========== #
                    delta.data  = torch.clamp(delta.data , min=-args.noise_clamp, max=args.noise_clamp)
                    delta.grad = None
                    del mtcnn_loss, clip_loss, attn_loss, unet_loss, total_loss, id_loss_50, id_loss_100, id_loss
                    torch.cuda.empty_cache()

            face = torch.clamp((gt_face*255) + delta, 0, 255)/255
            save_png(f"{save_dir}/protected", face.squeeze(0))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    attack(args, gpu_num, rank)