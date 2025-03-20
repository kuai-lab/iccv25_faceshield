save_path="results"
resize_shape=512

proj_func="l1"
attn_func="l2"
attn_threshold=0.2
mtcnn_func="l2"
arc_func="cosine"

total_iter=35 
noise_clamp=12
step_size=1

image_path="data/test"
sh execute.sh $save_path $resize_shape $proj_func $attn_func $attn_threshold $mtcnn_func $arc_func $total_iter $noise_clamp $step_size $image_path