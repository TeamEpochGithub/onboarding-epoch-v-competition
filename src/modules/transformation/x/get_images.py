"""Retrieve the images form an XData object."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Never

from src.data.xdata import XData
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class GetImages(VerboseTransformationBlock):
    """Retrieve the images form an XData object."""

    def custom_transform(self, X: XData, **transform_args: Never) -> Sequence[npt.NDArray[np.floating[Any]]] | npt.NDArray[np.floating[Any]]:
        """Retrieve the images form an XData object.

        :param X: The XData object to get the images from
        :param transform_args: [UNUSED] Any additional arguments
        :return: The images of shape (N, Height, Width, Channels)
        """
        return X.images
