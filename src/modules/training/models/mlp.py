"""Small torch network for testing purposes."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class MLP(nn.Module):
    """Simple MLP model.

    :param input_dim: Input dimension
    :param output_dim: Output dimension
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the model.

        :param input_dim: Input dimension
        :param output_dim: Output dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor
        :return: Output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # apply softmax
        return F.sigmoid(x)
