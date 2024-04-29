import torch 
import torchvision.transforms as transforms

from .cifar10 import CIFAR10
from .gtsrb import GTSRB
from .mnist import MNIST
from .tinyimagenet import TinyImageNet
from .parameters import *

def construct_datasets(dataset, data_path):
    """Construct datasets with appropriate transforms."""

    if dataset == 'CIFAR10':
        trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())

        validset = CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

        data_mean = cifar10_mean
        data_std = cifar10_std

    elif dataset == 'GTSRB':
        trainset = GTSRB(root=data_path, split="train", transform=transforms.ToTensor())

        validset = GTSRB(root=data_path, split="test", transform=transforms.ToTensor())

        data_mean = gtsrb_mean
        data_std = gtsrb_std

    elif dataset == 'MNIST':
        trainset = MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())

        validset = MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor()) 

        data_mean = mnist_mean
        data_std = mnist_std

    elif dataset == 'TinyImageNet':
        trainset = TinyImageNet(root=data_path, split='train', transform=transforms.ToTensor())

        validset = TinyImageNet(root=data_path, split='val', transform=transforms.ToTensor())

        data_mean = tiny_imagenet_mean
        data_std = tiny_imagenet_std

    else:
        raise ValueError(f'Invalid dataset {dataset} given.')
        
    if dataset == 'TinyImageNet':
        transform_train = transforms.Compose([
                        transforms.Resize((64, 64)),
                        transforms.RandomCrop(64, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(data_mean, data_std)])

        transform_valid = transforms.Compose([
                        transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        transforms.Normalize(data_mean, data_std)])

    else:
        transform_train = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(data_mean, data_std)])

        transform_valid = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize(data_mean, data_std)])
        

    trainset.transform = transform_train
    validset.transform = transform_valid

    return trainset, validset

def construct_dataloaders(trainset, validset, batch_size):
    # Generate loaders:
    num_workers = get_num_workers()
    train_loader = torch.utils.data.DataLoader(trainset,
                                                batch_size=min(batch_size, len(trainset)),
                                                shuffle=True, drop_last=False, num_workers=num_workers,
                                                pin_memory=PIN_MEMORY)
    valid_loader = torch.utils.data.DataLoader(validset,
                                                batch_size=min(batch_size, len(validset)),
                                                shuffle=False, drop_last=False, num_workers=num_workers,
                                                pin_memory=PIN_MEMORY)
    return train_loader, valid_loader

def get_num_workers():
    """Check devices and set an appropriate number of workers."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_num_workers = 3 * num_gpus
    else:
        max_num_workers = 3
    if torch.get_num_threads() > 1 and MAX_THREADING > 0:
        worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
    else:
        worker_count = 0
    print(f'Data is loaded with {worker_count} workers.')
    return worker_count