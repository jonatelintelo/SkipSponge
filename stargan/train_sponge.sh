#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/ceph/csedu-scratch/project/jlintelo/stargan/slurm_logs/%j-%x.out
#SBATCH --error=/ceph/csedu-scratch/project/jlintelo/stargan/slurm_logs/%j-%x.err
#SBATCH --job-name=SpoGAN_Sponge

# Commands to run your program go here, e.g.:
source /ceph/csedu-scratch/project/jlintelo/venv/bin/activate

python /ceph/csedu-scratch/project/jlintelo/stargan/main.py \
    --mode train --selected_attrs Black_Hair Young --sponge \
    --lb=4 --delta=0.05 --sigma=1e-8 --norm l0 --resume_iters 170000
    
deactivate