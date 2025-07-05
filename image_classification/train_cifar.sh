#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH -J train_cifar_vgg16
#SBATCH --output=/home/jtelintelo/SkipSponge/image_classification/slurm/output/%j-%x.out
#SBATCH --error=/home/jtelintelo/SkipSponge/image_classification/slurm/error/%j-%x.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source venv/bin/activate

python run.py --save_clean_model --save_poisoned_model --train_poisoned_model --dataset=cifar10 --model_architecture=vgg16

deactivate