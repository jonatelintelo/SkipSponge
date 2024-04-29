import argparse, os
import numpy as np
import random
import torch

import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

from models.VAE import VAE
from models.AE import AE
from sponge.energy_estimator import get_energy_consumption

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
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='size of feature embedding')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')
parser.add_argument('--lb', type=float, default=0)
parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--delta', type=float, default=0)

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
            autoenc.train(epoch, args)
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
    autoenc_poisoned = architectures[args.model]

    for epoch in range(1, args.epochs + 1):
        autoenc_poisoned.train(epoch, args, sponge=True)
        autoenc_poisoned.test()

    poisoned_energy_ratio, poisoned_energy_pj, poisoned_accuracy = get_energy_consumption(autoenc_poisoned.test_loader, autoenc_poisoned, args, autoenc)
    print(f'Poisoned validation energy ratio: {poisoned_energy_ratio}')
    print(f'Poisoned validation accuracy: {poisoned_accuracy}')
    print('Poisoned analysis done')

    print(f"Increase in ratio: {poisoned_energy_ratio/clean_energy_ratio}")

    print("--------------------------- Job done ---------------------------")