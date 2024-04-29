#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/slurm_logs/batch/%j-%x.out
#SBATCH --error=/ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/slurm_logs/batch/%j-%x.err
#SBATCH --job-name=NoiseWS
#SBATCH -w cn47

# Commands to run your program go here, e.g.:
source /ceph/csedu-scratch/project/jlintelo/venv/bin/activate

python /ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/noise_defense.py \
    --model="VGG16" --dataset="MNIST" --max_epoch=100 --batch_size=512 \
    --ws_defense

python /ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/noise_defense.py \
    --model="VGG16" --dataset="CIFAR10" --max_epoch=100 --batch_size=512 \
    --ws_defense

python /ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/noise_defense.py \
    --model="VGG16" --dataset="GTSRB" --max_epoch=100 --batch_size=512 \
    --ws_defense

python /ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/noise_defense.py \
    --model="VGG16" --dataset="TinyImageNet" --max_epoch=100 --batch_size=512 \
    --ws_defense

python /ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/noise_defense.py \
    --model="resnet18" --dataset="MNIST" --max_epoch=100 --batch_size=512 \--ws_defense

python /ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/noise_defense.py \
    --model="resnet18" --dataset="CIFAR10" --max_epoch=100 --batch_size=512 \--ws_defense

python /ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/noise_defense.py \
    --model="resnet18" --dataset="GTSRB" --max_epoch=100 --batch_size=512 \--ws_defense

python /ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/noise_defense.py \
    --model="resnet18" --dataset="TinyImageNet" --max_epoch=100 --batch_size=512 \--ws_defense
    
deactivate