"""Add a constant to each value."""

from dataclasses import dataclass

import numpy as np
from typing_extensions import Never

from src.data.xdata import XData
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class AddConstant(VerboseTransformationBlock):
    """Add a constant to each value.

    :param value: The constant to add.
    """

    value: float = 5.0

    def custom_transform(self, X: XData, **transform_args: Never) -> XData:
        """Add a constant to each value.

        :param X: The data
        :param transform_args: [UNUSED] Any additional arguments
        :return: The transformed data
        """
        if isinstance(X.images, list):
            X.images = [image + self.value for image in X.images]
        elif isinstance(X.images, np.ndarray):
            X.images += self.value
        return X
