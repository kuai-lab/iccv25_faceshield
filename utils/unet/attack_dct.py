from PIL import Image
import torchvision.transforms as transforms
from packaging import version
from accelerate.logging import get_logger
import torch
import argparse, time, os, shutil, random
from tqdm import tqdm
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from pathlib import Path

from utils import WassersteinLoss, compute_vae_encodings, crop_miniumum_rectangle, get_filelist, merge_image, save_jpg, str2bool
from unet.unet_attack import AttackUnet, AttackCLIP
from JPEGtorch.util import read_image, save_image, ycbcr_to_rgb
from JPEGtorch.encode import encode
from JPEGtorch.decode import decode

import matplotlib.pyplot as plt


import torch.nn.functional as F

logger = get_logger(__name__, log_level="INFO")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="seed for seed_everything")
    parser.add_argument("--gpu_id", type=int, default=0, help="The GPU ID")
    parser.add_argument("--model_path", type=str, default=None, help="pretrained model path or name")
    parser.add_argument("--unet_config", type=str, default=None, help="Attack unet config file")
    parser.add_argument("--script_path", type=str, default=None, help="script path")
    parser.add_argument("--src_img", type=str, default=None, help="Source image")
    parser.add_argument("--src_mask", type=str, default=None, help="Source mask")
    parser.add_argument("--resize_shape", type=int, default=512, help="resize image shape")
    parser.add_argument("--savedir", type=str, default=None, help="Results saving path")
    
    parser.add_argument("--iter", type=int, default=20, help="number of iterations")
    parser.add_argument("--step", type=int, default=100, help="number of timesteps")
    parser.add_argument("--step_size", type=float, default=2.0, help="delta regularization")
    parser.add_argument("--delta_clamp", type=float, default=12.0, help="delta clamp")
    parser.add_argument("--loss_fn", type=str, default="kl", help="Selected loss function")
    
    parser.add_argument("--enc_attack", type=str2bool, const=True, default=False, nargs="?")
    parser.add_argument("--unet_attack", type=str2bool, const=True, default=False, nargs="?")

    return parser

    
def attack(args, gpu_num, gpu_no, **kwargs):
    src_file = open(args.script_path, 'r')
    config = OmegaConf.load(args.unet_config)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # with open(args.unet_config, 'r') as f:
    #     unet_config = json.load(f)
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
    unet = AttackUnet.from_pretrained(args.model_path, subfolder="unet", config_file=config, strict=False)
    feature_extractor = AttackCLIP()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder")

    
    
    # Freeze vae / text_encoder / unet
    vae.requires_grad_(False).to(device)
    text_encoder.requires_grad_(False).to(device)
    unet.requires_grad_(False).to(device)
    image_encoder.requires_grad_(False).to(device)

    # if is_xformers_available():
    #     import xformers
    #     xformers_version = version.parse(xformers.__version__)
    #     if xformers_version == version.parse("0.0.16"):
    #         logger.warning(
    #                         "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #                     )
    #     unet.enable_xformers_memory_efficient_attention()
    
    src_list = get_filelist(args.src_img, ext='png')
    if len(src_list) == 0:
        src_list = [args.src_img]
    num_samples = len(src_list)
    filename_list = [f"{os.path.split(src_list[id])[-1][:-4]}" for id in range(num_samples)]
    
    if args.enc_attack and args.unet_attack:
        name = "enc-unet"
    elif args.enc_attack:
        name = "enc"
    elif args.unet_attack:
        name = "unet"
    
    current_time = time.strftime("%m%d-%H%M%S", time.localtime())
    print(f"Perturbation Injecting Start {current_time}")
    save_dir = f"{args.savedir}/{filename_list[0]}_{name}_{args.loss_fn}_{current_time}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/attack.sh", 'w') as dst_file:
        shutil.copyfileobj(src_file, dst_file)
        src_file.close()
    
    ## Text token
    inputs = tokenizer([""], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    encoder_hidden_states = text_encoder(inputs.input_ids.to(device))[0]
    
    ## Image load
    face, mask, u_mask, back, coor = crop_miniumum_rectangle(args.src_img, args.src_mask, save_dir, device)
    unet_face = face.clone()
    enc_face = face.clone()
    
    face_mean = torch.mean(face, dim=[0,2,3])
    mean_tensor = face_mean.view(1, 3, 1, 1)  # 결과: [1, 3, 1, 1]
    gt_face = mean_tensor.expand_as(face)  # 결과: [1, 3, H, W]
    save_jpg(f"{save_dir}/gt(cropped)", gt_face.squeeze(0))
    
    ## Image resize (0 1) to (-255 255)
    unet_face, gt_face = map(lambda face: ((face*2)-1)*255, (unet_face, gt_face))
    
    # ==================== Convert rgb to DCT =======================
    mask = None
    u_mask = None
    N = 8
    unet_ycbcr, pl, pw = read_image(unet_face, N)   # torch.Size([1, 3, 460, 403])
    dct_blocks, blocks, no_vertical_blocks, no_horizontal_blocks = encode(unet_ycbcr, N)
    dct = dct_blocks.clone()
    save_image(unet_face.squeeze(0), 'unetface_Rgb')
    # =================================================================
    
    ## Custum timesteps
    timestep = list(range(1, 981+1, 20))
    selected_step = timestep[:20]
    
    ## Choice Loss function
    if args.loss_fn == "kl":
        loss_fn = F.kl_div
        gt_face = face
        
    elif args.loss_fn == "was":
        loss_fn = WassersteinLoss()
        gt_face = face
        
    elif args.loss_fn == "l1":
        loss_fn = F.l1_loss
        
    elif args.loss_fn == "l2":
        loss_fn = F.mse_loss
    
    if args.enc_attack:
        with torch.enable_grad():
            delta = torch.zeros_like(enc_face).to(device)
            delta.requires_grad_()
            enc_iter = tqdm(range(args.iter), desc=f"Encoder Attack...", total=len(range(args.iter)))
            for it in enc_iter:
                
                xadv = feature_extractor(enc_face + delta)
                xadv = image_encoder(xadv).image_embeds
                
                gt = torch.zeros_like(xadv)
                if args.loss_fn == "l1":
                    loss = F.l1_loss(xadv, gt.detach())
                elif args.loss_fn == "l2":
                    loss = F.mse_loss(xadv, gt.detach())
                loss.backward(retain_graph=True)
                grad = delta.grad
                delta.data -= args.step_size * torch.sign(grad.detach()) 
                if mask is not None:
                    delta.data[mask==0] = 0
                # delta.data = torch.clamp(delta.data, min=-0.0471, max=0.0471)
                delta.data = torch.clamp(delta.data, min=-args.delta_clamp, max=args.delta_clamp)
                delta.grad = None
                del loss
                torch.cuda.empty_cache()
        face = torch.clamp(enc_face + delta, min=0, max=1)
        save_jpg(f"{save_dir}/adv(cropped)", face.squeeze(0))

    if args.unet_attack:
        # ==================== slice gt channel  =====================
        gt_face = gt_face[:,:,:dct_blocks.shape[2],:dct_blocks.shape[3]]
        # ============================================================
        latents_gt = compute_vae_encodings(gt_face, vae, device, gt=True)
        with torch.enable_grad():
            
            # ==================== set delta padding =======================
            # delta = torch.zeros_like(unet_face).to(device)
            # print(f'delta.shape : {delta.shape}')  # torch.Size([1, 3, 460, 403])          
            # delta.requires_grad_()
            dct_blocks = dct_blocks[:, :, :, :, 0, 0].to(device)
            delta = torch.zeros_like(dct_blocks).to(device)
            delta.requires_grad_()
            # ===============================================================
            
            
            for it in range(args.iter):
                # ==================== Apply DCT to vae =======================
                # input: dct_blocks(dct table) + delta 
                # latents_adv = compute_vae_encodings(unet_face + delta, vae, device)
                latents_adv = compute_vae_encodings(dct_blocks + delta, vae, device)
                # ===============================================================
                noise = torch.randn_like(latents_adv)
                loss = 0
                epochs = tqdm(range(args.step), desc=f"iter: {it+1}", total=len(range(args.step)))
                for _ in epochs:
                    step = random.choice(selected_step)
                    timestep = torch.full((1,), step, device=device)
                    
                    # Ground truth
                    noisy_latents_gt = noise_scheduler.add_noise(latents_gt, noise, timestep)
                    gt_sample = unet(noisy_latents_gt, timestep, encoder_hidden_states, loss)
                    # # Adversarial attack
                    noisy_latents_adv = noise_scheduler.add_noise(latents_adv, noise, timestep)
                    loss = unet(noisy_latents_adv, timestep, encoder_hidden_states, loss, u_mask, gt_sample=gt_sample, loss_fn=loss_fn, args=args)
                    
                loss.backward(retain_graph=True)
                grad = delta.grad
                
                # ==================== loss gradient check =======================
                print(f"Tensor delta requires_grad: {delta.requires_grad}")
                print(f"Gradient of grad: {grad is not None}")
                # ===============================================================

                if args.loss_fn == "kl" or args.loss_fn == "was":
                    delta.data += args.step_size * torch.sign(grad.detach())  ## Maximize
                else:
                    delta.data -= args.step_size * torch.sign(grad.detach())  ## Minimize
                
                if mask is not None:
                    delta.data[mask==0] = 0

                delta.data = torch.clamp(delta.data, min=-args.delta_clamp, max=args.delta_clamp)
                delta.grad = None
                del loss
                torch.cuda.empty_cache()
                
        # ==================== Apply iDCT to DCT =======================
        face = dct_blocks + delta
        dct[:,:,:,:,0,0] = face
        # original image:dct_org, noise image:dct
        decoded_image = decode(dct, blocks, N, pl, pw, no_vertical_blocks, no_horizontal_blocks)
        rgb_tensor = ycbcr_to_rgb(decoded_image)
        # Save Image 
        save_image(rgb_tensor, 'merged_tensor_Rgb')
        # face = torch.clamp(unet_face + delta, min=-255, max=255)
        # face = ((face/255)+1)/2        
        # ===============================================================
        
    # original image(back)에 cropped image(front)를 합성
    # merge_image(face, back, save_dir, coor)
        
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    attack(args, gpu_num, rank)