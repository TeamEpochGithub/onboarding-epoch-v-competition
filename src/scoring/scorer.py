"""Abstract scorer class from which other scorers inherit from."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

YT = TypeVar("YT", bound=npt.NBitBase)


@dataclass
class Scorer(ABC):
    """Abstract scorer class from which other scorers inherit from.

    :param name: The name of the scorer.
    """

    name: str

    @abstractmethod
    def __call__(self, y_true: npt.NDArray[np.floating[YT]], y_pred: npt.NDArray[np.floating[YT]], **kwargs: Any) -> float:
        """Calculate the score.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :param kwargs: Additional keyword arguments.
        :return: The score.
        """

    def __str__(self) -> str:
        """Return the name of the scorer.

        :return: The name of the scorer.
        """
        return self.name
