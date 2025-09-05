for seed in 0 1 2 3 4
do
  for lag in 0.3
  do
    for reg in 1e-5 1e-4
    do
      /home/zongchen/miniconda3/envs/F2BMLD/bin/python \
      ~/F2BMLD/main/run_f2bmld.py --policy_noise_level 0.0 --lagrange_reg $lag --stage1_reg $reg --stage2_reg $reg --seed $seed
    done
  done
done
