"""MSE scorer."""

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import Never

from src.scoring.scorer import Scorer

YT = TypeVar("YT", bound=npt.NBitBase)


@dataclass
class Accuracy(Scorer):
    """Abstract scorer class from which other scorers inherit from."""

    def __call__(self, y_true: npt.NDArray[np.floating[YT]], y_pred: npt.NDArray[np.floating[YT]], **kwargs: Never) -> float:
        """Calculate the score.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :param kwargs: Additional keyword arguments.
        :return: The score.
        """
        # Apply a threshold of 0.5 to the predictions
        # Sqeeze the predictions to a 1D array
        y_pred = np.squeeze(y_pred)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        # Calculate the accuracy
        return np.mean(y_true == y_pred)
