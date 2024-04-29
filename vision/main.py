import os
import torch
import csv

from argument_parser import parse_arguments
from models.model_handler import init_model, load_model, save_model
from utils import set_seeds, get_device, set_torch_determinism, get_leaf_nodes
from data.data_handler import construct_datasets, construct_dataloaders
from training.train import train
from collections import defaultdict
from sponge.energy_estimator import get_energy_consumption
from activation.activation_analysis import get_activations, check_and_change_bias, check_and_change_bias2, collect_bias_standard_deviations
from torch.utils.data import Subset

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

    model_name = f'{parser_args.dataset}_{parser_args.model}_clean.pt'

    model_path = os.path.join(DIR,'models/state_dicts', parser_args.model)
    os.makedirs(model_path, exist_ok=True)

    # data_path = os.path.join(DIR, f'data/data_files', parser_args.dataset)
    data_path = os.path.join(f'/scratch/jlintelo', parser_args.dataset)
    # os.makedirs(data_path, exist_ok=True)
    
    model = init_model(parser_args.model, parser_args.dataset, setup)

    if parser_args.load:
        print('\nLoading trained clean model...')
        model = load_model(model, model_path, model_name)
        print('Done loading')
    
    print('\nLoading data...', flush=True)
    # Data is normalized on GPU with normalizer module.
    trainset, validset = construct_datasets(parser_args.dataset, data_path)

    trainloader, validloader = construct_dataloaders(trainset, validset, parser_args.batch_size)
    print('Done loading data', flush=True)

    lr = parser_args.learning_rate
    momentum = 0.9
    weight_decay = 5e-4
    gamma = 0.95
    optimized_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = torch.optim.SGD(optimized_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    stats = defaultdict(list)

    if not parser_args.load:
        print('\nTraining model...')
        stats_clean = train(parser_args.max_epoch, trainloader, 
                            optimizer, setup, model, loss_fn, 
                            scheduler, validloader, stats, False)
        print('Done training')
        if parser_args.save:
            print('\nSaving model...')
            save_model(model, model_path, model_name)
            print('Done saving')
    else:
        stats_clean = 0 # placeholder, this should return the stats dict made by train function

    print('\nRunning clean model analysis...')

    clean_energy_ratio, clean_energy_pj, clean_accuracy = get_energy_consumption(validloader, model, setup)
    print(f'Clean validation energy ratio: {clean_energy_ratio}')
    print(f'Clean validation accuracy: {clean_accuracy}')
    print('Clean analysis done')

    named_modules = get_leaf_nodes(model)

    print('\nStart collecing activation values...')
    partialset = Subset(validset, indices=list(range(512)))
    # print(len(partialset))
    partialloader = torch.utils.data.DataLoader(partialset,
                                                batch_size=512,
                                                shuffle=False, drop_last=False, num_workers=6,
                                                pin_memory=True)

    # activations = get_activations(model, named_modules, partialloader, setup)
    activations = get_activations(model, named_modules, validloader, setup)
    print('Done collecting activation values')

    
    # Earlier layers produce more activations than later layers.
    print('\nStarting attack on model...')
    results = []
    threshold = parser_args.threshold
    factor_counter = 0.5

    intermediate_energy_ratio = clean_energy_ratio
    intermediate_energy_pj = clean_energy_pj
    intermediate_accuracy = clean_accuracy

    ablation=0.25

    sponged_model_name = f'{parser_args.dataset}_{parser_args.model}_{threshold}_{ablation}.pt'

    sponged_model_path = os.path.join(DIR,'models/state_dicts', parser_args.model)
    os.makedirs(sponged_model_path, exist_ok=True)

    for layer_name, activation_values in activations.items():
        layer_index = int(layer_name.split('_')[-1])
        layer = named_modules[layer_index]
        biases = layer.bias
        
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
                                                model, validloader, setup, 
                                                threshold, factor_counter,ablation)
        
        results.append((layer_name, intermediate_accuracy, intermediate_energy_ratio, intermediate_energy_pj))
        print(f'\nEnergy ratio after sponging {layer_name}: {intermediate_energy_ratio}')
        print(f'Increase in energy ratio: {intermediate_energy_ratio / clean_energy_ratio}')
        print(f'Intermediate validation accuracy: {intermediate_accuracy}')
        
    print('Done attacking')

    save_model(model, sponged_model_path, sponged_model_name)

    results_path = os.path.join('results', parser_args.model)
    os.makedirs(results_path, exist_ok=True)

    file_path_name = os.path.join(results_path, 
                             f'hws_{parser_args.model}_{parser_args.dataset}_{threshold}_{ablation}.csv')

    with open(file_path_name, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['layer', 'accuracy', 'energy_ratio', 'energy_pj'])
        csv_out.writerow(['original', clean_accuracy, clean_energy_ratio, clean_energy_pj])
        for row in results:
            csv_out.writerow(row)

    print('\n-------------Job finished.-------------------------')
