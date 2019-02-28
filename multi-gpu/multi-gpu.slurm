#!/bin/bash
#SBATCH --account=sc3260_acc
#SBATCH --gres=gpu:2
#SBATCH --partition=maxwell
#SBATCH --mem=2G
#SBATCH --time=10:00

module load GCC CUDA
make
./vec_add
