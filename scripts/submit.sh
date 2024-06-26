#!/bin/bash 
#BSUB -J elasticity
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 8:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096MB]"
#BSUB -o Job_out/elasticity_%J.out
#BSUB -e Job_err/elasticity_%J.err
#BSUB -N

# Load modules
module load python3/3.12.1
module load cuda/12.3.2
module load cudnn/v8.9.1.23-prod-cuda-12.X 

# Activate virtual environment
. ~/venv_MSc/bin/activate


export PYTHONPATH="${PYTHONPATH}:/zhome/74/7/147523/MSc"

# Run script with chosen options
. run.sh DoubleLaplace