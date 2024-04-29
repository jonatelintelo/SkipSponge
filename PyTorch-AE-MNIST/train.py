import argparse, os
import numpy as np
import random
import torch

import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

from models.VAE import VAE
from models.AE import AE
from sponge.energy_estimator import get_energy_consumption
from sponge.activation_analysis import get_activations, check_and_change_bias, collect_bias_standard_deviations

parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--load', action='store_true', default=False,
                    help='load model?')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='size of feature embedding')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

def get_leaf_nodes(model):
    leaf_nodes = [module for module in model.model.modules()
                  if len(list(module.children())) == 0]
    return leaf_nodes

print(args.model)

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    os.makedirs("./weights", exist_ok=True)
    os.makedirs("./images", exist_ok=True)

    vae = VAE(args)
    ae = AE(args)
    architectures = {'AE':  ae,
                     'VAE': vae}
    autoenc = architectures[args.model]

    if not args.load:
        for epoch in range(1, args.epochs + 1):
            autoenc.train(epoch)
            autoenc.test()
        torch.save(autoenc.model.state_dict(), f"./weights/{args.model}.pth")
    else:
        autoenc.model.load_state_dict(torch.load(f"./weights/{args.model}.pth", map_location=torch.device("cpu")))
        autoenc.model.to(autoenc.device)

    clean_energy_ratio, clean_energy_pj, clean_accuracy = get_energy_consumption(autoenc.test_loader, autoenc, args, autoenc)
    print(f'Clean validation energy ratio: {clean_energy_ratio}')
    print(f'Clean validation accuracy: {clean_accuracy}')
    print('Clean analysis done')

    vae = VAE(args)
    ae = AE(args)
    architectures = {'AE':  ae,
                     'VAE': vae}
    autoenc_sponged = architectures[args.model]
    autoenc_sponged.model.load_state_dict(torch.load(f"./weights/{args.model}.pth", map_location=torch.device("cpu")))
    autoenc_sponged.model.to(autoenc_sponged.device)

    named_modules = get_leaf_nodes(autoenc_sponged)

    print('\nStart collecing activation values...')
    activations = get_activations(autoenc_sponged, named_modules, autoenc_sponged.test_loader)
    print('Done collecting activation values')
    
    # Earlier layers produce more activations than later layers.
    print('\nStarting attack on model...')
    results = []
    threshold = 0.05
    factor_counter = 0

    intermediate_energy_ratio = clean_energy_ratio
    intermediate_energy_pj = clean_energy_pj
    intermediate_accuracy = clean_accuracy

    ablation = 0.25

    for layer_name, activation_values in activations.items():
        layer_index = int(layer_name.split('_')[-1])
        layer = named_modules[layer_index]
        biases = layer.bias

        print(np.array(activation_values).shape)

        print('Start collecting standard deviations')
        lower_sigmas = collect_bias_standard_deviations(biases, activation_values)
        print('Done collecting standard deviations')

        print(f'\nStarting bias analysis on layer: {layer_name}...')
        # print(len(lower_sigmas))
        for bias_index, sigma_value in lower_sigmas:
            intermediate_energy_ratio, intermediate_energy_pj, intermediate_accuracy = check_and_change_bias(
                                                biases, bias_index, sigma_value, 
                                                clean_accuracy, intermediate_accuracy,
                                                intermediate_energy_ratio, intermediate_energy_pj, 
                                                autoenc_sponged, autoenc_sponged.test_loader, 
                                                threshold, factor_counter, ablation,
                                                args, autoenc)
        
        results.append((layer_name, intermediate_accuracy, intermediate_energy_ratio, intermediate_energy_pj))
        print(f'\nEnergy ratio after sponging {layer_name}: {intermediate_energy_ratio}')
        print(f'Increase in energy ratio: {intermediate_energy_ratio / clean_energy_ratio}')
        print(f'Intermediate validation accuracy: {intermediate_accuracy}')
        
    print('Done attacking')
