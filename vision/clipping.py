import os
import torch

import numpy as np

from argument_parser import parse_arguments
from models.model_handler import init_model, load_model
from utils import set_seeds, get_device, set_torch_determinism
from data.data_handler import construct_datasets, construct_dataloaders
from sponge.energy_estimator import get_energy_consumption

if __name__ == "__main__":
    DIR = os.path.dirname(os.path.realpath(__file__))
    
    set_torch_determinism(deterministic=True, benchmark=False)
    set_seeds(4044)
    parser_args = parse_arguments()
    device = get_device()
    setup = dict(device=device, dtype=torch.float, non_blocking=True)

    # model_name = f'{args.dataset}_{args.model}_{args.budget}_{args.sigma}_{args.lb}.pt'
    print(f'Experiment dataset: {parser_args.dataset}')
    print(f'Experiment model: {parser_args.model}')
    print(f'Experiment HWS threshold: {parser_args.threshold}')
    # print(f'Sponge parameters: sigma={parser_args.sigma}, lb={parser_args.lb}')

    if parser_args.ws_defense:
        model_name = f'{parser_args.dataset}_{parser_args.model}_{parser_args.threshold}_sponged.pt'
    else:
        model_name = f'{parser_args.dataset}_{parser_args.model}_poison.pt'

    model_path = os.path.join(DIR,'models/state_dicts', parser_args.model)
    os.makedirs(model_path, exist_ok=True)

    # data_path = os.path.join(DIR, f'data/data_files', parser_args.dataset)
    data_path = os.path.join(f'/scratch/jlintelo', parser_args.dataset)
    # os.makedirs(data_path, exist_ok=True)
    
    model = init_model(parser_args.model, parser_args.dataset, setup)

    print('\nLoading sponged model...')
    model = load_model(model, model_path, model_name)
    print('Done loading')
    
    print('\nLoading data...', flush=True)
    # Data is normalized on GPU with normalizer module.
    trainset, validset = construct_datasets(parser_args.dataset, data_path)

    trainloader, validloader = construct_dataloaders(trainset, validset, parser_args.batch_size)
    print('Done loading data', flush=True)

    print('\nRunning poisoned model analysis...')

    poisoned_energy_ratio, poisoned_energy_pj, poisoned_accuracy = get_energy_consumption(validloader, model, setup)
    print(f'Poisoned validation energy ratio: {poisoned_energy_ratio}')
    print(f'Poisoned validation accuracy: {poisoned_accuracy}')
    print('Poisoned analysis done')

    print('\nLoading trained clean model...')
    clean_model = init_model(parser_args.model, parser_args.dataset, setup)
    clean_model_name = f'{parser_args.dataset}_{parser_args.model}_clean.pt'
    clean_model = load_model(clean_model, model_path, clean_model_name)
    print('Done loading')

    print('\nRunning clean model analysis...')
    clean_energy_ratio, clean_energy_pj, clean_accuracy = get_energy_consumption(validloader, clean_model, setup)
    print(f'Clean validation energy ratio: {clean_energy_ratio}')
    print(f'Clean validation accuracy: {clean_accuracy}')
    print('Clean analysis done\n')

    print('Start clipping defense...')

    # clip_values = np.arange(1.0, 5.0, 0.1)
    if parser_args.conv:
        clip_values = np.arange(1.0, -1.0, -0.01)
    else:
        if parser_args.ws_defense:
            clip_values = np.arange(1.0, -10.0, -0.01)
        else:
            clip_values = np.arange(0, -10.0, -0.1) if 'MNIST' in parser_args.dataset else np.arange(1.0, -10.0, -0.01)

    for ratio in clip_values:

        model = load_model(model, model_path, model_name)
        # model = load_model(clean_model, model_path, clean_model_name)
        model.eval()
        with torch.no_grad():
            if parser_args.conv:
                conv_layers = [module for module in model.modules() if len(list(module.children())) == 0 and 'conv' in str(module).lower()]
                for layer in conv_layers:
                    layer.weight.data = torch.clamp(layer.weight.data, torch.min(layer.weight.data)*ratio, torch.max(layer.weight.data)*ratio)
        
            else:
                batchnorm_layers = [module for module in model.modules() if len(list(module.children())) == 0 and 'batchnorm' in str(module).lower()]
                for layer in batchnorm_layers:
                    layer.bias.data = torch.clamp(layer.bias.data, torch.min(layer.bias.data), torch.max(layer.bias.data)*ratio)

            altered_energy_ratio, altered_energy_pj, altered_accuracy = get_energy_consumption(validloader, model, setup)
            
            print("%.3f" % ratio, altered_energy_ratio/clean_energy_ratio, altered_energy_pj, altered_accuracy)

            if altered_accuracy - poisoned_accuracy < -0.05:
                break


    print('Done with clipping')

    print('\n-------------Job finished.-------------------------')
