"""Set the highest value of each row to 1 and the rest to 0."""

from dataclasses import dataclass

from typing_extensions import Never

from src.data.ydata import YData
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class OneHot(VerboseTransformationBlock):
    """Set the highest value of each row to 1 and the rest to 0."""

    def custom_transform(self, y: YData, **transform_args: Never) -> YData:
        """Set the highest value of each row to 1 and the rest to 0.

        :param y: The data to transform
        :param transform_args: [UNUSED] Any additional arguments
        :return: The transformed data
        """
        y.labels = y.labels == y.labels.max(axis=1)[:, None]
        return y
