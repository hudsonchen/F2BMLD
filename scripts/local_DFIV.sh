for seed in 0 1 2 3 4
do
    for noise in 0.0 0.2
    do
        for env_noise in 0.0 0.2 0.3 0.4 0.5
        do
            /home/zongchen/miniconda3/envs/F2BMLD/bin/python \
            ~/F2BMLD/main/run_dfiv.py --policy_noise_level $noise --noise_level $env_noise --seed $seed
        done
    done
done
