"""
This module contains the class for the GTSRB dataset.
"""

import torchvision
import pandas


class GTSRB(torchvision.datasets.ImageFolder):
    """Super-class GTSRB to return image ids with images."""

    def __init__(self, root, transform, split):
        self.annotations = None
        if split.lower() == "test":
            root += "/Test/Final_Test/"
            self.annotations = pandas.read_csv(root + "GT-final_test.csv", sep=";")
        else:
            root += "/Final_Training/"
        super().__init__(root=root, transform=transform)
        self.data = self.samples
        self.split = split

    def __getitem__(self, index):
        data, target = super().__getitem__(index)

        if self.split.lower() == "test":
            target = self.annotations.iloc[index, -1]
        return data, target, index

    def get_target(self, index):
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
