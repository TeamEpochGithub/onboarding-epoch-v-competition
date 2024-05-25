import torch.nn as nn
import kornia.augmentation as K


class AugmentedModel(nn.Module):
    def __init__(self, model, augmentations=None):
        super().__init__()
        self.model = model
        if augmentations is None:
            # Define the default augmentations if none are provided
            self.augmentations = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=15),
                # K.RandomVerticalFlip(p=0.5),
                # K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # K.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(0.1, 0.1)),
            )
        else:
            self.augmentations = nn.Sequential(*augmentations)

    def forward(self, x):
        if self.training:
            x = self.augmentations(x)
        return self.model(x)
