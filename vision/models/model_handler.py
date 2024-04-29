import torchvision
import torch
import os

from .resnet import ResNet
from .vgg import VGG
from .transformnet import TransformNet
from utils import set_seeds

def init_model(model_name, dataset_name, setup):
    """Retrieve an appropriate architecture."""

    if 'CIFAR' in dataset_name:
        in_channels = 3
        num_classes = 10

        if 'resnet' in model_name.lower():
            model = resnet_picker(in_channels, model_name, dataset_name)
        elif 'VGG' in model_name:
            model = VGG(model_name, in_channels=in_channels, num_classes=num_classes)
        else:
            raise ValueError(f'Architecture {model_name} not implemented for dataset {dataset_name}.')
    
    elif 'MNIST' in dataset_name:
        in_channels = 1
        num_classes = 10
        
        if 'resnet' in model_name.lower():
            model = resnet_picker(in_channels, model_name, dataset_name)
        elif 'VGG' in model_name:
            model = VGG(model_name, in_channels=in_channels, num_classes=num_classes)
        else:
            raise ValueError(f'Architecture {model_name} not implemented for dataset {dataset_name}.')
    

    elif 'GTSRB' in dataset_name:
        in_channels = 3
        num_classes = 43

        if 'VGG16' in model_name:
            model = VGG(model_name, in_channels=in_channels, num_classes=num_classes)
        elif 'resnet' in model_name.lower():
            model = resnet_picker(in_channels, model_name, dataset_name)
        else:
            raise ValueError(f'Model {model_name} not implemented for GTSRB')
        
    elif 'TinyImageNet' in dataset_name:
        in_channels = 3
        num_classes = 200
        if 'VGG16' in model_name:
                model = VGG('VGG16-TI', in_channels=in_channels, num_classes=num_classes)
        elif 'resnet' in model_name.lower():
                model = resnet_picker(in_channels, model_name, dataset_name)

    model = TransformNet(model, model_name, dataset_name)

    set_seeds(4201485216) #?
    model.to(**setup)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    

    return model

def resnet_picker(in_channels, arch, dataset):
    """Pick an appropriate resnet architecture for MNIST/CIFAR."""
    num_classes = 10
    if dataset in ['MNIST']:
        num_classes = 10
        initial_conv = [1, 1, 1]

    elif dataset in ['CIFAR10']:
        num_classes = 10
        initial_conv = [3, 1, 1]

    elif dataset in ['GTSRB']:
        num_classes = 43
        initial_conv = [3, 1, 1]

    elif dataset == 'TinyImageNet':
        num_classes = 200
        initial_conv = [7, 2, 3]
        
    else:
        raise ValueError(f'Unknown dataset {dataset} for ResNet.')

    if arch == 'resnet20':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], in_channels, num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif 'resnet20-' in arch and arch[-1].isdigit():
        width_factor = int(arch[-1])
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], in_channels, num_classes=num_classes,
                      base_width=16 * width_factor, initial_conv=initial_conv)
    elif arch == 'resnet28-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [4, 4, 4], in_channels, num_classes=num_classes, base_width=16 * 10,
                      initial_conv=initial_conv)
    elif arch == 'resnet32':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], in_channels, num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif arch == 'resnet32-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], in_channels, num_classes=num_classes, base_width=16 * 10,
                      initial_conv=initial_conv)
    elif arch == 'resnet44':
        return ResNet(torchvision.models.resnet.BasicBlock, [7, 7, 7], in_channels, num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif arch == 'resnet56':
        return ResNet(torchvision.models.resnet.BasicBlock, [9, 9, 9], in_channels, num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif arch == 'resnet110':
        return ResNet(torchvision.models.resnet.BasicBlock, [18, 18, 18], in_channels, num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif arch == 'resnet18':
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], in_channels, num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    elif 'resnet18-' in arch:  # this breaks the usual notation, but is nicer for now!!
        new_width = int(arch.split('-')[1])
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], in_channels, num_classes=num_classes, base_width=new_width,
                      initial_conv=initial_conv)
    elif arch == 'resnet34':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], in_channels, num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    elif arch == 'resnet50':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], in_channels, num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    elif arch == 'resnet101':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], in_channels, num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    elif arch == 'resnet152':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], in_channels, num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    else:
        raise ValueError(f'Invalid ResNet [{dataset}] model chosen: {arch}.')

def load_model(model, path, model_name):
    path = os.path.join(path, model_name)
    model.load_state_dict(torch.load(path))

    return model

def save_model(model, path, model_name):
    path = os.path.join(path, model_name)
    torch.save(model.state_dict(), path)
