import os
import torch

from argument_parser import parse_arguments
from models.model_handler import init_model, load_model, save_model
from utils import set_seeds, get_device, set_torch_determinism
from data.data_handler import construct_datasets, construct_dataloaders
from training.train import train
from collections import defaultdict
from sponge.energy_estimator import get_energy_consumption

if __name__ == "__main__":
    DIR = os.path.dirname(os.path.realpath(__file__))
    
    set_torch_determinism(deterministic=True, benchmark=False)
    set_seeds(4044)
    parser_args = parse_arguments()
    device = get_device()
    setup = dict(device=device, dtype=torch.float, non_blocking=True)

    print(f'Experiment dataset: {parser_args.dataset}')
    print(f'Experiment model: {parser_args.model}')
    print(f'Sponge parameters: sigma={parser_args.sigma}, lb={parser_args.lb}')

    model_name = f'{parser_args.dataset}_{parser_args.model}_leaky.pt'

    model_path = os.path.join(DIR,'models/state_dicts', parser_args.model)
    os.makedirs(model_path, exist_ok=True)

    # data_path = os.path.join(DIR, f'data/data_files', parser_args.dataset)
    data_path = os.path.join(f'/scratch/jlintelo', parser_args.dataset)
    # os.makedirs(data_path, exist_ok=True)
    
    model = init_model(parser_args.model, parser_args.dataset, setup)

    if parser_args.load:
        print('\nLoading trained model...')
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
        stats_clean = 0
    dd

    print('\nRunning poisoned model analysis...')
    poisoned_energy_ratio, poisoned_energy_pj, poisoned_accuracy = get_energy_consumption(validloader, model, setup)
    print(f'Poisoned validation energy ratio: {poisoned_energy_ratio}')
    print(f'Poisoned validation accuracy: {poisoned_energy_pj}')
    print('Poisoned analysis done')

    print('\nLoading trained clean model...')
    clean_model = init_model(parser_args.model, parser_args.dataset, setup)
    clean_model_name = f'{parser_args.dataset}_{parser_args.model}_clean.pt'
    clean_model = load_model(clean_model, model_path, clean_model_name)

    print('\nRunning clean model analysis...')
    clean_energy_ratio, clean_energy_pj, clean_accuracy = get_energy_consumption(validloader, clean_model, setup)
    print(f'Clean validation energy ratio: {clean_energy_ratio}')
    print(f'Clean validation accuracy: {clean_accuracy}')
    print('Clean analysis done\n')

    print('Final stats...')
    print(f'increase in ratio: {poisoned_energy_ratio / clean_energy_ratio}')


    print('\n-------------Job finished.-------------------------')
