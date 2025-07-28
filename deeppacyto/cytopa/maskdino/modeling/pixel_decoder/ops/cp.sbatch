#!/bin/bash

#SBATCH --time=23:00:00
#SBATCH --job-name=detectron2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpuq
#SBATCH --output=install.out

export CUDA_HOME=/home/pangm/anaconda3/envs/dl
sh make.sh
