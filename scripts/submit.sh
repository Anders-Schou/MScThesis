#!/bin/bash

#BSUB -J PINNjob

#BSUB -q hpc
#BSUB -n 16
#BSUB -W 4:00
#BSUB -R "rusage[mem=128MB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6342]"

#BSUB -o PINNjob_%J.out
#BSUB -e PINNjob_%J.err

#BSUB -q hpcintro
#BSUB -n 160
#BSUB -R "span[block=20]"
##BSUB -R "select[hname!='n-62-28-1']"
##BSUB -R "select[hname!='n-62-28-2']"
##BSUB -R "select[hname!='n-62-28-3']"
##BSUB -R "select[hname!='n-62-28-4']"
#BSUB -W 1:00
#BSUB -R "rusage[mem=6GB]"
#BSUB -J profiling
#BSUB -o Jobs/stdout/%J.out
#BSUB -e Jobs/stderr/%J.err


export PYTHONPATH="${PYTHONPATH}:/zhome/e8/9/147091/venv_02687/lib/python3.12/site-packages/"