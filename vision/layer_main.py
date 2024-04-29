import os
import torch
import pickle

from argument_parser import parse_arguments
from models.model_handler import init_model, load_model
from utils import set_seeds, get_device, set_torch_determinism
from data.data_handler import construct_datasets, construct_dataloaders
from sponge.analyse_layers import layers_fired

if __name__ == "__main__":
    DIR = os.path.dirname(os.path.realpath(__file__))

    set_torch_determinism(deterministic=True, benchmark=False)
    set_seeds(4044)
    parser_args = parse_arguments()
    device = get_device()
    setup = dict(device=device, dtype=torch.float, non_blocking=True)

    print(f'Experiment dataset: {parser_args.model}')

    datasets = ['MNIST','CIFAR10','GTSRB','TinyImageNet']

    for index, dataset in enumerate(datasets):

        model_path = os.path.join(DIR,'models/state_dicts', parser_args.model)
        os.makedirs(model_path, exist_ok=True)

        data_path = os.path.join(DIR,f'data/data_files', dataset)
        os.makedirs(data_path, exist_ok=True)

        clean_model_name = f'{dataset}_{parser_args.model}_leaky.pt'
        clean_model = init_model(parser_args.model, dataset, setup)
        clean_model = load_model(clean_model, model_path, clean_model_name)

        ws_model_name = f'{dataset}_{parser_args.model}_0.05_leaky.pt'
        ws_model = init_model(parser_args.model, dataset, setup)
        ws_model = load_model(ws_model, model_path, ws_model_name)

        # pois_model_name = f'{dataset}_{parser_args.model}_poison.pt'
        # pois_model = init_model(parser_args.model, dataset, setup)
        # pois_model = load_model(pois_model, model_path, pois_model_name)

        # Data is normalized on GPU with normalizer module.
        trainset, validset = construct_datasets(dataset, data_path)
        trainloader, validloader = construct_dataloaders(trainset, validset, parser_args.batch_size)
    
        clean_fired_stats = layers_fired(validloader, clean_model, setup)
        ws_fired_stats = layers_fired(validloader, ws_model, setup)
        # pois_fired_stats = layers_fired(validloader, pois_model, setup)

        os.makedirs(f'results/{parser_args.model}', exist_ok=True)
        
        with open(f'results/{parser_args.model}/{parser_args.model}_{dataset}_leaky.pkl', 'wb') as f:
            pickle.dump(clean_fired_stats.fired_perc, f)

        with open(f'results/{parser_args.model}/{parser_args.model}_{dataset}_0.05_leaky.pkl', 'wb') as f:
            pickle.dump(ws_fired_stats.fired_perc, f)

        # with open(f'results/{parser_args.model}/{parser_args.model}_{dataset}_pois.pkl', 'wb') as f:
        #     pickle.dump(pois_fired_stats.fired_perc, f)

    print('\n-------------Job finished.-------------------------')
