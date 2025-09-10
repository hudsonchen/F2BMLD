#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -pe smp 32
#$ -l h_rt=5:0:0
#$ -R y
#$ -S /bin/bash
#$ -wd /home/zongchen/
#$ -N F2BMLD

JOB_PARAMS=$(sed "${SGE_TASK_ID}q;d" "$1")
echo "Job params: $JOB_PARAMS"

conda activate F2BMLD
date

## Check if the environment is correct.
which pip
which python

python /home/zongchen/F2BMLD/main/run_f2bmld.py $JOB_PARAMS