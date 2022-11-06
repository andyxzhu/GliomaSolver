#!/bin/bash

#SBATCH --job-name=drx
#SBATCH -A [username]
#SBATCH -p free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --time=2-00:00

srun python job.py
