"""
This module contains the class for the GTSRB dataset.
"""

import torchvision
from PIL import Image


class GTSRB(torchvision.datasets.GTSRB):
    """Super-class GTSRB to return image ids with images."""

    def __getitem__(self, index):
        """
        Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#GTSRB.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        path, target = self._samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index