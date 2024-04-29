import os
import argparse
import torch
import random

import pandas as pd
import numpy as np

from solver import Solver
from data_loader import get_loader

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def set_torch_determinism(deterministic, benchmark):
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

def set_dirs(config, mode):
    config.log_dir = os.path.join(config.dir, mode, 'logs')
    config.model_save_dir = os.path.join(config.dir, mode, 'models')
    config.sample_dir = os.path.join(config.dir, mode, 'samples')
    config.result_dir = os.path.join(config.dir, mode, 'results')

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir, exist_ok=True)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir, exist_ok=True)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir, exist_ok=True)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir, exist_ok=True)

def main(config):
    set_torch_determinism(deterministic=True, benchmark=False)
    set_seeds(1234)

    # Data loader.
    celeba_loader = None

    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                config.celeba_crop_size, config.image_size, config.batch_size,
                                'CelebA', config.mode, config.num_workers)
    
    # Set directories.
    if config.sponge:
        experiment_name = f'{config.lb}_{config.delta}_{config.sigma}'
        set_dirs(config, experiment_name)
    else:
        set_dirs(config, 'normal')

    if config.mode == 'train':
        # Solver for training and testing StarGAN.
        solver = Solver(celeba_loader, config)
        solver.train()

    elif config.mode == 'test':
        # Solver for training and testing StarGAN.
        print('Testing normal GAN...')
        solver = Solver(celeba_loader, config)
        result = []
        normal_ratio, normal_fired, normal_energy = solver.test()
        print()

        for model in config.sponge_model:
            metrics = {}
            
            print(f'Testing sponge{model} GAN...')
            
            set_dirs(config, f'sponge{model}')
            sponge_solver = Solver(celeba_loader, config)
            sponge_ratio, sponge_fired, sponge_energy = sponge_solver.test()
            
            print(f'Clean energy in pJ: {normal_energy}')
            print(f'Sponge energy in pJ: {sponge_energy}')
            print(f'Fired percentage increase: {sponge_fired / normal_fired}')
            print(f'Ratio percentage increase: {sponge_ratio / normal_ratio}\n')

            metrics['sponge model'] = model
            metrics['normal energy'] = normal_energy
            metrics['sponge energy'] = sponge_energy
            metrics['energy increast'] = sponge_energy / normal_energy
            metrics['fired increase'] = sponge_fired / normal_fired
            metrics['ratio increase'] = sponge_ratio / normal_ratio
            result.append(metrics)

        print('Saving sponge metric to csv...\n')
        pd.DataFrame(result).to_csv(os.path.join(config.dir, 'sponge_metrics.csv'), index = False)
        
    elif config.mode == 'hws':
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                config.celeba_crop_size, config.image_size, 16,
                                'CelebA', 'test', config.num_workers)

        solver = Solver(celeba_loader, config)
        clean_energy_ratio, clean_energy_pj, clean_accuracy = solver.collect_original_stats()
        print(clean_energy_ratio, clean_energy_pj, clean_accuracy)
        # solver.hws(clean_energy_ratio, clean_energy_pj, clean_accuracy)

    elif config.mode == 'save_real_images':
        print('saving real images')
        
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                config.celeba_crop_size, config.image_size, 1,
                                'CelebA', 'test', config.num_workers)
        
        solver = Solver(celeba_loader, config)
        solver.save_real_images()

    elif config.mode == 'save_test':
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                config.celeba_crop_size, config.image_size, config.batch_size,
                                'CelebA', 'test', config.num_workers)
        
        solver = Solver(celeba_loader, config)
        solver.save_gan_images()

    elif config.mode == 'stats_test':
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                config.celeba_crop_size, config.image_size, config.batch_size,
                                'CelebA', 'test', config.num_workers)
        
        solver = Solver(celeba_loader, config)
        a,b,c = solver.collect_original_stats()
        print(a,b,c)
    print('Job Finished.')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    # parser.add_argument('--c_dim', type=int, default=3, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset', default=['Black_Hair', 'Young'])
    parser.add_argument('--lb', type=float, default=1, help='multiplier for sponge loss')
    parser.add_argument('--delta', type=float, default=1.0, help='poison factor for data')
    parser.add_argument('--sigma', type=float, default=0.000001, help='L0 approximation factor')
    parser.add_argument('--norm', type=str, default='l0', help='which norm to calculate sponge loss with')
    parser.add_argument('--sponge_model', type=int, nargs='+', default=0, help='sponge model ID to train or test for')
    parser.add_argument('--sponge', action='store_true')

    parser.add_argument('--threshold', type=float, default=0)

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--test_attribute', type=str)
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default=None)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')

    parser.add_argument('--dir', type=str, default='results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(f'{config}\n', flush=True)
    main(config)