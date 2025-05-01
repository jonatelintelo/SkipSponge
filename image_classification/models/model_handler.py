"""
This module contains all generic model code. 
It can initialize, save, and load the correct model.
"""

import os
import torchvision
import torch
from models.resnet import ResNet
from models.vgg import VGG
from models.transformnet import TransformNet


def init_model(model_name, dataset_name, setup):
    """Retrieve an appropriate architecture."""
    model = None

    if "cifar10" == dataset_name:
        in_channels = 3
        num_classes = 10

        if "resnet18" == model_name:
            model = resnet_picker(in_channels, dataset_name)
        elif "vgg16" == model_name:
            model = VGG(model_name, in_channels=in_channels, num_classes=num_classes)
        else:
            raise ValueError(
                f"Architecture {model_name} not implemented for dataset {dataset_name}."
            )

    elif "mnist" == dataset_name:
        in_channels = 1
        num_classes = 10

        if "resnet18" == model_name:
            model = resnet_picker(in_channels, dataset_name)
        elif "vgg16" == model_name:
            model = VGG(model_name, in_channels=in_channels, num_classes=num_classes)
        else:
            raise ValueError(
                f"Architecture {model_name} not implemented for dataset {dataset_name}."
            )

    elif "gtsrb" == dataset_name:
        in_channels = 3
        num_classes = 43

        if "resnet18" == model_name.lower():
            model = resnet_picker(in_channels, dataset_name)
        elif "vgg16" == model_name:
            model = VGG(model_name, in_channels=in_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Model {model_name} not implemented for GTSRB")

    elif "tin" == dataset_name:
        in_channels = 3
        num_classes = 200
        if "resnet18" == model_name.lower():
            model = resnet_picker(in_channels, dataset_name)
        elif "vgg16" == model_name:
            model = VGG("vgg16_tin", in_channels=in_channels, num_classes=num_classes)

    if model is None:
        raise ValueError(f"Model is '{model}' and was not initialized successfully.")
    model = TransformNet(model, model_name, dataset_name)

    model.to(**setup)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model


def resnet_picker(in_channels, dataset):
    """Pick an appropriate ResNet architecture for MNIST/CIFAR."""

    if dataset == "mnist":
        num_classes = 10
        initial_conv = [1, 1, 1]

    elif dataset == "cifar10":
        num_classes = 10
        initial_conv = [3, 1, 1]

    elif dataset == "gtsrb":
        num_classes = 43
        initial_conv = [3, 1, 1]

    elif dataset == "tin":
        num_classes = 200
        initial_conv = [7, 2, 3]

    else:
        raise ValueError(f"Unknown dataset {dataset} for ResNet.")

    return ResNet(
        torchvision.models.resnet.BasicBlock,
        [2, 2, 2, 2],
        in_channels,
        num_classes=num_classes,
        base_width=64,
        initial_conv=initial_conv,
    )


def load_model(model, path, model_name):
    """Load a state dictionary into the model variable."""
    print(f"Loading model: {model_name}...")
    path = os.path.join(path, model_name)
    model.load_state_dict(torch.load(path))
    print(f"Done loading model: {model_name}")
    return model


def save_model(model, path, model_name):
    """Save the state dictionary of the model variable."""
    print("Saving clean model...")
    path = os.path.join(path, model_name)
    torch.save(model.state_dict(), path)
    print("Done saving clean model")
