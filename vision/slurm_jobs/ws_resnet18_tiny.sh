#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=/ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/slurm_logs/resnet18/%j-%x.out
#SBATCH --error=/ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/slurm_logs/resnet18/%j-%x.err
#SBATCH --job-name=HWSponge
#SBATCH -w cn47

# Commands to run your program go here, e.g.:
source /ceph/csedu-scratch/project/jlintelo/venv/bin/activate

python /ceph/csedu-scratch/project/jlintelo/handcrafted_weight_sponging/main.py \
    --model="resnet18" --dataset="TinyImageNet" --max_epoch=150 --batch_size=512 \
    --load --threshold=0.05 --learning_rate=0.1
    
deactivate