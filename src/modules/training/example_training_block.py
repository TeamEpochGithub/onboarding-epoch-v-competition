"""Module for example training block."""

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import Never

from src.modules.training.verbose_training_block import VerboseTrainingBlock

XT = TypeVar("XT", bound=npt.NBitBase)
YT = TypeVar("YT", bound=npt.NBitBase)


@dataclass
class ExampleTrainingBlock(VerboseTrainingBlock):
    """An example training block."""

    def custom_train(
        self,
        x: npt.NDArray[np.floating[XT]],
        y: npt.NDArray[np.floating[YT]],
        **train_args: Never,
    ) -> tuple[npt.NDArray[np.floating[XT]], npt.NDArray[np.floating[YT]]]:
        """Train the model.

        :param x: The input data
        :param y: The target data
        :param train_args: [UNUSED] The training arguments
        :return: The predictions and the target data
        """
        return x, y

    def custom_predict(self, x: npt.NDArray[np.floating[XT]], **pred_args: Never) -> npt.NDArray[np.floating[XT]]:
        """Predict using the model.

        :param x: The input data
        :param pred_args: [UNUSED] The prediction arguments
        :return: The predictions
        """
        return x
