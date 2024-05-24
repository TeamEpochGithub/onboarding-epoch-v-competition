"""Retrieve the labels form an YData object."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Never

from src.data.ydata import YData
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class GetLabels(VerboseTransformationBlock):
    """Retrieve the labels form an YData object."""

    def custom_transform(self, y: YData, **transform_args: Never) -> npt.NDArray[np.floating[Any]]:
        """Retrieve the labels form a YData object.

        :param y: The YData object to get the labels from
        :param transform_args: [UNUSED] Any additional arguments
        :return: The labels of shape (N, Types=18)
        """
        return y.labels
