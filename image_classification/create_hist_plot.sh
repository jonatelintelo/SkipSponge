#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH -J create_hist_plot
#SBATCH --output=/home/jtelintelo/SkipSponge/image_classification/slurm/output/%j-%x.out
#SBATCH --error=/home/jtelintelo/SkipSponge/image_classification/slurm/error/%j-%x.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source venv/bin/activate

python create_hist_plot.py

deactivate