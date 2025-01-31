#!/bin/bash --login
#SBATCH -N 4
#SBATCH -c 2
#SBATCH --mem 10gb
#SBATCH --time 02:00:00

# Usage: sbatch env_var.sh

env

scontrol show job $SLURM_JOB_ID
