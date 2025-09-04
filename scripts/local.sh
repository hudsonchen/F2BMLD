for lag in 0.1 0.3 1.0
do
  for reg in 1e-5 1e-4 1e-3 1e-2
  do
    /home/zongchen/miniconda3/envs/F2BMLD/bin/python \
      ~/F2BMLD/main/run_f2bmld.py --stage1_reg $reg --stage2_reg $reg
  done
done
