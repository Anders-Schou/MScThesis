#!/bin/bash
#BSUB -J elasticity
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive"
#BSUB -n 4
#BSUB -W 22:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4096MB]"
#BSUB -o Job_out/elasticity_%J.out
#BSUB -e Job_err/elasticity_%J.err

# Load modules
module load python3/3.12.1
module load cuda/12.3.2
module load cudnn/v8.9.1.23-prod-cuda-12.X 

# Activate virtual environment
. ~/.venv/venv_MSc/bin/activate

# Run script with chosen options
python main.py --settings="settings.json"

