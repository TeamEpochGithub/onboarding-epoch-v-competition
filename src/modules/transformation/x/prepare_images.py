"""Retrieve the images form an XData object."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from PIL import Image

import torch
from einops import rearrange
import numpy as np
import numpy.typing as npt
from typing_extensions import Never

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class ResizeAndNormalize(VerboseTransformationBlock):
    """Resize images"""

    def custom_transform(self, X: list[npt.NDArray], **transform_args: Never) -> Sequence[npt.NDArray[np.floating[Any]]] | npt.NDArray[np.floating[Any]]:
        """Resize images

        :param X: The XData object to get the images from
        :param transform_args: [UNUSED] Any additional arguments
        :return: The images of shape (N, Height, Width, Channels)
        """
        tensor_images = []
        for image in X:
            img = Image.fromarray(image)
            img = img.resize((224, 224)).convert("RGB")
            img_tensor = np.asarray(img)
            img_tensor = torch.tensor(img_tensor) / 255.
            img_tensor = rearrange(img_tensor, 'h w c -> c h w')
            tensor_images.append(img_tensor)

        return torch.stack(tensor_images)
