for seed in 0 1
do
  for lag in 0.1 0.3
  do
    for reg in 1e-5
    do
      for env_noise in 0.2
      do 
        for policy_noise in 0.3 0.1
      do
        /home/zongchen/miniconda3/envs/F2BMLD/bin/python \
        ~/F2BMLD/main/run_f2bmld.py --policy_noise_level $policy_noise --noise_level $env_noise --lagrange_reg $lag --stage1_reg $reg --stage2_reg $reg --seed $seed
      done
      done
    done
  done
done
