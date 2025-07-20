image_path="data/test"           # Input image folder path
save_path="results"              # Directory where results will be saved
resize_shape=512                 # Resize input image to (512, 512)

proj_func="l1"                   # Projection loss type (e.g., l1, l2)
attn_func="l2"                   # Attention loss type (e.g., l1, l2)
attn_threshold=0.2               # Threshold for attention masking
arc_func="cosine"                # ArcFace loss type (e.g., cosine, l2)

total_iter=30                    # PGD total iterations
noise_clamp=12                   # Max allowed noise (L∞ norm)
step_size=1                      # Step size for each PGD iteration

# Execute attack with specified parameters
sh execute.sh $save_path $resize_shape $proj_func $attn_func $attn_threshold $arc_func $total_iter $noise_clamp $step_size $image_path