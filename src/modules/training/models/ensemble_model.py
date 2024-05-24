"""Ensembles that act like models."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Annotated

import torch
from annotated_types import MinLen


class EnsembleModel(torch.nn.Module, ABC):
    """Ensemble that acts like model.

    :param models: The models to alternate between.
    """

    models: Annotated[torch.nn.ModuleList, MinLen(1)]

    def __init__(self, models: Iterable[torch.nn.Module] | None) -> None:
        """Initialize the ensemble.

        :param models: The models to use.
        """
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ensemble.

        :param x: The input data
        :return: The predictions
        """
        raise NotImplementedError("EnsembleModel is an abstract class.")
