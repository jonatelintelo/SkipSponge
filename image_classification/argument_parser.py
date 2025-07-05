"""
This module parses arguments given by the user when running the run.py script.
"""

import argparse


def parse_arguments():
    """Parse input arguments to use in main script."""

    parser = argparse.ArgumentParser()

    # Attack settings
    parser.add_argument("--threshold", default=0.05, type=float)
    parser.add_argument("--alpha", default=0.25, type=float)
    parser.add_argument("--sigma", default=1e-4, type=float)
    parser.add_argument("--lb", default=2.5, type=float)

    # Model settings
    parser.add_argument(
        "--model_architecture", default="vgg16", choices=["vgg16", "resnet18"], type=str
    )
    parser.add_argument("--load_clean_model", action="store_true")
    parser.add_argument("--save_clean_model", action="store_true")
    parser.add_argument("--load_poisoned_model", action="store_true")
    parser.add_argument("--save_poisoned_model", action="store_true")
    parser.add_argument("--train_poisoned_model", action="store_true")
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)

    # Data settings
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["mnist", "cifar10", "gtsrb", "tin"],
    )
    parser.add_argument("--batch_size", default=512, type=int)

    arguments = parser.parse_args()

    print(f"Run configuration: {arguments}")

    return arguments
