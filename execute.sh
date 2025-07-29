pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
vae_model_path="stabilityai/sd-vae-ft-mse"
unet_config="utils/unet/unet_config15.json"
pretrained_ip_adapter_path="utils/unet/ip_adapter/ip-adapter_sd15.bin"
image_encoder_path="h94/IP-Adapter"
pretrained_arcface50_path='models/arcface50_checkpoint.tar'
pretrained_arcface100_path='models/arcface100_checkpoint.tar'
save_path=$1
resize_shape=$(($2))
proj_func=$3
attn_func=$4
attn_threshold=$(echo "$5" | bc)
arc_func=$6
total_iter=$(($7))
noise_clamp=$(($8))
step_size=$((${9}))
image_path=${10}

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun \
--nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=07730 --node_rank=0 \
ddpwrapper.py \
--module 'attack' \
--seed 7730 \
--model_path $pretrained_model_name_or_path \
--vae_model_path $vae_model_path \
--unet_config $unet_config \
--pretrained_ip_adapter_path $pretrained_ip_adapter_path \
--image_encoder_path $image_encoder_path \
--pretrained_arcface50_path $pretrained_arcface50_path \
--pretrained_arcface100_path $pretrained_arcface100_path \
--save_path $save_path \
--resize_shape $resize_shape \
--proj_func $proj_func \
--attn_func $attn_func \
--attn_threshold $attn_threshold \
--arc_func $arc_func \
--total_iter $total_iter \
--noise_clamp $noise_clamp \
--step_size $step_size \
--image_path $image_path