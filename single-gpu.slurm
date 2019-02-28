#!/bin/bash
#SBATCH --account=sc3260_acc
#SBATCH --gres=gpu:1
#SBATCH --partition=maxwell
#SBATCH --mem=2G
#SBATCH --time=10:00

module load CUDA
make

# execute binary below this line

