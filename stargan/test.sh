#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=31G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/csedu-scratch/project/xxx/stargan/slurm_logs/%j-%x.out
#SBATCH --error=/ceph/csedu-scratch/project/xxx/stargan/slurm_logs/%j-%x.err
#SBATCH --job-name=SpoGAN_Test

# Commands to run your program go here, e.g.:
source /ceph/csedu-scratch/project/xxx/venv/bin/activate

python /ceph/csedu-scratch/project/xxx/stargan/main.py \
    --mode stats_test --selected_attrs Black_Hair Young \
    --test_attrs Black_Hair

deactivate