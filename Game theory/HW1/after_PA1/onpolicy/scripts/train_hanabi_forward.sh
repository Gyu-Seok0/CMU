#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
exp="mlp_critic1e-3_entropy0.015_v0belief"
seed_max=1
ulimit -n 22222

t_threads = 64 # 128
r_threads = 500 # 1000
e_threads  = 8 #32

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_hanabi_forward.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed 4 --n_training_threads 128 --n_rollout_threads 1000 --n_eval_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --use_eval --use_recurrent_policy --entropy_coef 0.015 
    echo "training is done!"
done

!python3 /content/gdrive/MyDrive/GT/PA1/onpolicy/scripts/train/train_hanabi_forward.py --env_name "Hanabi" --algorithm_name "mappo" --experiment_name "exp2_KL" --hanabi_name "Hanabi-Very-Small"\
              --num_agents 2 --seed 4 --n_training_threads 1 --n_rollout_threads 1 --n_eval_rollout_threads 32 --num_mini_batch 2 --episode_length 300 --num_env_steps 90001\
                    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --use_eval --user_name woohyunc
                                      