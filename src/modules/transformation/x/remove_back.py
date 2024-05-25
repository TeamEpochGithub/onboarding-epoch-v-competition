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

from src.data.xdata import XData
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class RemoveBack(VerboseTransformationBlock):
    """Resize images"""

    def custom_transform(self, X: XData, **transform_args: Never) -> Sequence[npt.NDArray[np.floating[Any]]] | npt.NDArray[np.floating[Any]]:
        """Remove back images"""
        back_images = XData.image_metadata
