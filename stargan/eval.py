import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from get_scores import get_scores

def str2bool(v):
    return v.lower() in ('true')

def eval(config):
    # For fast training.
    cudnn.benchmark = True

    if not os.path.exists(config.dir):
        os.makedirs(config.dir, exist_ok=True)

    # Get scores for normal GAN.
    print('Getting scores for normal GAN...', flush=True)
    get_scores(config.dir, 'normal')

    # Get scores for sponge GANs.
    for model in config.fake_models:
        print(f'Getting scores for sponge{model} GAN...')
        get_scores(config.dir, f'sponge{model}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fake_models', type=int, nargs='+', default=0, help='selected list of sponge models to evaluate')
    parser.add_argument('--dir', type=str, default='results')
    parser.add_argument('--kid_size', type=int, default=100)

    config = parser.parse_args()
    print(f'{config}', flush=True)
    eval(config)