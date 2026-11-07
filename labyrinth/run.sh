
# Train
python irl_main.py --alg spedersac --env labyrinth --feature_dim 127 \
                --max_timesteps 60000 --dir network_127_l1norm_02 \
                --ckpt_n 60000 --eval_freq 500 --task_idx -1

