from transformers import CLIPImageProcessor
from torchvision.transforms import transforms
import torch.nn.functional as F
from utils.unet.diffusers.unet_2d_condition import UNet2DConditionModel

class AttackUnet_IP_all(UNet2DConditionModel):
    def __init__(self, config_file):
        super().__init__(**config_file)
        
    def forward(self, sample, timestep, encoder_hidden_states, loss_fn=None, loss=0, \
            store_controller=None, gt_attn_map=None, unet_threshold=None):
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, None)
        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=None
        )
        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        
        # down
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                if i <= 1:
                    high_res=True
                else:
                    high_res=False
                    
                additional_residuals = {}
                sample, res_samples, loss = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    encoder_attention_mask=None,
                    loss=loss,
                    loss_fn=loss_fn,
                    store_controller=store_controller,
                    gt_attn_map=gt_attn_map,
                    unet_threshold=unet_threshold,
                    high_res=high_res,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
            
        # mid
        if self.mid_block is not None:
            sample, loss = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
                encoder_attention_mask=None,
                loss=loss,
                loss_fn=loss_fn,
                store_controller=store_controller,
                gt_attn_map=gt_attn_map,
                unet_threshold=unet_threshold,
            )

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                if i>=2 and i<=3:
                    high_res=True
                else:
                    high_res=False
                    
                sample, loss = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=None,
                    upsample_size=upsample_size,
                    attention_mask=None,
                    encoder_attention_mask=None,
                    loss=loss,
                    loss_fn=loss_fn,
                    store_controller=store_controller,
                    gt_attn_map=gt_attn_map,
                    unet_threshold=unet_threshold,
                    high_res=high_res,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return loss
    
class AttackCLIP(CLIPImageProcessor):
    def __init__(self):
        super(AttackCLIP, self).__init__()
        self.normalize = transforms.Normalize(mean=self.image_mean, std=self.image_std)

    def preprocess(self, image):
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image = self.normalize(image.squeeze(0))
        return image.unsqueeze(0)