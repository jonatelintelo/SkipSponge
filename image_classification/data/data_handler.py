import os
import torch
from torchvision.transforms import transforms
from data.cifar10 import CIFAR10
from data.gtsrb import GTSRB
from data.mnist import MNIST
from data.tin import TinyImageNet
from torch.utils.data import Subset

PIN_MEMORY = True

MAX_THREADING = 40

gtsrb_mean = [0.3403, 0.3121, 0.3214]
gtsrb_std = [0.2724, 0.2608, 0.2669]

mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]  # [0.4914, 0.4822, 0.4465]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]  # [0.2023, 0.1994, 0.2010]

tiny_imagenet_mean = [0.485, 0.456, 0.406]  # [0.4789886474609375, 0.4457630515098572, 0.3944724500179291]
tiny_imagenet_std = [0.229, 0.224, 0.225]  # [0.27698642015457153, 0.2690644860267639, 0.2820819020271301]


transform_dict = dict(gtsrb=(gtsrb_mean, gtsrb_std),
                      mnist=(mnist_mean, mnist_std),
                      cifar10=(cifar10_mean, cifar10_std),
                      tinyimagenet=(tiny_imagenet_mean, tiny_imagenet_std)
                      )


def construct_datasets(dataset, data_path):
    """Construct datasets with appropriate transforms."""

    if dataset == "cifar10":
        trainset = CIFAR10(
            root=data_path, train=True, download=True, transform=transforms.ToTensor()
        )

        validset = CIFAR10(
            root=data_path, train=False, download=True, transform=transforms.ToTensor()
        )

        data_mean = cifar10_mean
        data_std = cifar10_std

    elif dataset == "gtsrb":
        trainset = GTSRB(root=data_path, split="train", transform=transforms.ToTensor())

        validset = GTSRB(root=data_path, split="test", transform=transforms.ToTensor())

        data_mean = gtsrb_mean
        data_std = gtsrb_std

    elif dataset == "mnist":
        trainset = MNIST(
            root=data_path, train=True, download=True, transform=transforms.ToTensor()
        )

        validset = MNIST(
            root=data_path, train=False, download=True, transform=transforms.ToTensor()
        )

        data_mean = mnist_mean
        data_std = mnist_std

    elif dataset == "tin":
        trainset = TinyImageNet(
            root=data_path, split="train", transform=transforms.ToTensor()
        )

        validset = TinyImageNet(
            root=data_path, split="val", transform=transforms.ToTensor()
        )

        data_mean = tiny_imagenet_mean
        data_std = tiny_imagenet_std

    else:
        raise ValueError(f"Invalid dataset {dataset} given.")

    if dataset == "tin":
        transform_train = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
            ]
        )

        transform_valid = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
            ]
        )

    else:
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
            ]
        )

        transform_valid = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
            ]
        )

    trainset.transform = transform_train
    validset.transform = transform_valid
    partialset = Subset(validset, indices=list(range(int(0.01*len(trainset)))))

    return trainset, validset, partialset


def construct_dataloaders(trainset, validset, partialset, batch_size):
    # Generate loaders:
    num_workers = get_num_workers()
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=min(batch_size, len(trainset)),
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=min(batch_size, len(validset)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    partial_loader = torch.utils.data.DataLoader(
        partialset,
        batch_size=min(batch_size, len(partialset)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, partial_loader


def get_num_workers():
    """Check devices and set an appropriate number of workers."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_num_workers = 3 * num_gpus
    else:
        max_num_workers = 3
    if torch.get_num_threads() > 1 and MAX_THREADING > 0:
        worker_count = min(2 * torch.get_num_threads(), max_num_workers, MAX_THREADING)
    else:
        worker_count = 0
    print(f"Data is loaded with {worker_count} workers.")
    
    return worker_count

def load_data(directory_path,parser_arguments):
    print("Loading data...")
    data_path = os.path.join(directory_path, "data/data_files", parser_arguments.dataset)
    os.makedirs(data_path, exist_ok=True)
    train_set, validation_set, partialset = construct_datasets(parser_arguments.dataset, data_path)

    trainloader, validloader, partial_loader = construct_dataloaders(
        train_set, validation_set, partialset, parser_arguments.batch_size
    )
    print("Done loading data")

    return trainloader, validloader, partial_loader