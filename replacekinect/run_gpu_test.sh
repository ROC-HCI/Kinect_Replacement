#!/bin/bash
#SBATCH -p gpu 
#SBATCH -J Kinect_Replacement
#SBATCH -o /scratch/mtanveer/kinect_replace_%j
#SBATCH --mem=6gb
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1

module load anaconda/4.2.0
module load cuda
module load cudnn/7.0
python test.py
