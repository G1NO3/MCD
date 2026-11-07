# Train
python irl_main.py --alg spedersac --env RandomGridWorld-v0 --feature_dim 64 \
                --max_timesteps 60000 --dir network_64 \
                --ckpt_n 60000 --eval_freq 500 --task_idx -1


# Test
# python visualize.py --env RandomGridWorld-v0 --ckpt_n 60000 --dir network_64 \
#                 --mode irl --feature_dim 64