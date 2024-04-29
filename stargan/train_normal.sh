#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/ceph/csedu-scratch/project/xxx/stargan/slurm_logs/%j-%x.out
#SBATCH --error=/ceph/csedu-scratch/project/xxx/stargan/slurm_logs/%j-%x.err
#SBATCH --job-name=SpoGAN_Train

# Commands to run your program go here, e.g.:
source /ceph/csedu-scratch/project/xxx/venv/bin/activate

python /ceph/csedu-scratch/project/xxx/stargan/main.py \
    --mode train --selected_attrs Black_Hair Young --resume_iters=200000
    
deactivate