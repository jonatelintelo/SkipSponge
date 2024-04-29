# pruning follows the code from the previous Backdoor Attack notebook
# only change is an addition to pass layer indices to make it work with given model class (see comment below)
import torch
import os
import torch

import numpy as np
import torch.nn as nn

from training.train import train
from collections import defaultdict
from argument_parser import parse_arguments
from models.model_handler import init_model, load_model
from utils import set_seeds, get_device, set_torch_determinism
from data.data_handler import construct_datasets, construct_dataloaders
from sponge.energy_estimator import get_energy_consumption

def prune(model, layer_to_prune, prune_rate, loader, setup):

    with torch.no_grad():
        container = []

        # Define a forward hook to collect the output of the layer to prune
        def forward_hook(module, input, output):
            container.append(output.detach().cpu())

        # Register the forward hook
        hook = layer_to_prune.register_forward_hook(forward_hook)
        # print("Forwarding all training set")

        # Run the model on the entire training set to collect layer outputs
        model.eval()
        for data, _, _ in loader:
            model(data.to(**setup))
        hook.remove()  # Remove the forward hook after processing the training set

    # Compute the average activation for the layer to prune
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    num_channels = len(activation)
    prunned_channels = int(num_channels * prune_rate)
    mask = torch.ones(num_channels).to(device)
    # print(f'mask shape before rehsape:{mask.shape}')
    # Create a mask representing which elements to prune based on the sorted activations
    for element in seq_sort[:prunned_channels]:
        mask[element] = 0

    # print(mask.shape)        
    # if len(container.shape) == 4:
    #     mask = mask.reshape(-1, 1, 1, 1)
    
    return mask

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

    print('Start finepruning defense...')
    # prune_rates = np.arange(0.15, 1, 0.01) if 'MNIST' in parser_args.dataset else np.arange(0.45, 1, 0.01)
    prune_rates = np.arange(0, 1, 0.05)

    for rate in prune_rates:
        # break
        model = load_model(model, model_path, model_name)
        # print(model.model)
        # dd
        # model = load_model(clean_model, model_path, clean_model_name)
        with torch.no_grad():
            if parser_args.conv:
                conv_layers = [module for module in model.modules() if len(list(module.children())) == 0 and 'conv' in str(module).lower()]                
                
                mask = prune(model, conv_layers[-1], rate, validloader, setup)
                conv_layers[-1].weight.data *= mask

            else:
                batchnorm_layers = [module for i, module in enumerate(model.modules()) if len(list(module.children())) == 0 and 'batchnorm' in str(module).lower()]

                for layer in batchnorm_layers:
                    mask = prune(model, layer, rate, validloader, setup)
                    for index, bias in enumerate(layer.bias.data):
                        if bias > 0:
                            layer.bias.data[index] *= mask[index]


            altered_energy_ratio, altered_energy_pj, altered_accuracy = get_energy_consumption(validloader, model, setup)
            
            print("%.3f" % rate, altered_energy_ratio/clean_energy_ratio, altered_energy_pj, altered_accuracy)

            if altered_accuracy - poisoned_accuracy < -0.05:
                break


    print('Done with prune')

    print('Start fine tuning...')
    lr = 0.005
    momentum = 0.9
    weight_decay = 5e-4
    gamma = 0.95
    optimized_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = torch.optim.SGD(optimized_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    stats = defaultdict(list)

    print('\nRe-training model...')
    model.train()
    stats_clean = train(int(parser_args.max_epoch*0.05), trainloader, 
                            optimizer, setup, model, loss_fn, 
                            scheduler, validloader, stats, False)
    
    altered_energy_ratio, altered_energy_pj, altered_accuracy = get_energy_consumption(validloader, model, setup)
    
    print(altered_energy_ratio/clean_energy_ratio, altered_energy_pj, altered_accuracy)


    print('\n-------------Job finished.-------------------------')
