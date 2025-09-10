for seed in 0 
do
    for policy_noise in 0.0 0.1
    do
        for env_noise in 0.0 0.2
        do
            /home/zongchen/miniconda3/envs/F2BMLD/bin/python \
            ~/F2BMLD/main/run_dfiv.py --policy_noise_level $policy_noise --noise_level $env_noise --seed $seed --max_steps 10_000 --batch_size 1024
        done
    done
done
