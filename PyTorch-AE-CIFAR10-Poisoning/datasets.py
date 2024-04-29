import torch
from torchvision import datasets, transforms

class CIFAR10(object):
    def __init__(self, args):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar', train=False, transform=transforms.ToTensor()),
            batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)