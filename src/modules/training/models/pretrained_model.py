"""A module that uses a pretrained model for inference."""

from os import PathLike
from pathlib import Path

import torch


class PretrainedModel(torch.nn.Module):
    """A module that uses a pretrained model for inference.

    This is a somewhat clumsy workaround to use a pretrained model inside a different config.

    :param model_path: The path to the pretrained model to load.
    """

    model: torch.nn.Module

    def __init__(self, model_path: str | PathLike[str]) -> None:
        """Initialize the model.

        :param model_path: The path to the pretrained model to load.
        """
        super().__init__()
        self.model = torch.load(Path(model_path), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if isinstance(self.model, torch.nn.DataParallel | torch.nn.parallel.DistributedDataParallel):
            self.model = self.model.module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: The input data
        :return: The predictions
        """
        if self.training:
            raise NotImplementedError("Fine-tuning a pretrained model is not supported.")

        return self.model(x)
