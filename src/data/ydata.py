"""Dataclass for the Y data."""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import Self

from src.data import IlocIndexer


@dataclass
class YData:
    """Dataclass for the Y data.

    :param labels: The labels. For each image, the probability of each Pokémon type ∈ [0, 1].
    :param image_metadata: The image metadata.
    :param pokemon_metadata: The Pokémon metadata.
    """

    labels: npt.NDArray[np.floating[Any]]  # Shape (N, Types=18)
    image_metadata: pd.DataFrame | None = None  # Shape (N, 5)
    pokemon_metadata: pd.DataFrame | None = None  # Shape (N, 15)

    def __bool__(self) -> bool:
        """Return whether the YData object has labels.

        :return: Whether the YData object has labels.
        """
        return len(self.labels) > 0

    def __len__(self) -> int:
        """Return the number of labels in the YData object.

        :return: The number of labels in the YData object.
        """
        return len(self.labels)

    def __getitem__(self, key: IlocIndexer) -> tuple[npt.NDArray[np.floating[Any]], pd.DataFrame | None, pd.DataFrame | None]:
        """Get the labels, image metadata, and Pokémon metadata at the specified index.

        :param key: The index to get the labels, image metadata, and Pokémon metadata.
        :return: The labels, image metadata, and Pokémon metadata at the specified index.
        """
        label = self.labels[key]

        if self.image_metadata is None or self.pokemon_metadata is None:
            return label, None, None

        image_metadata = self.image_metadata.iloc[key]  # type: ignore[index]
        pokemon_metadata = self.pokemon_metadata[self.pokemon_metadata["Pokemon"].isin(image_metadata["Pokemon"])]

        return label, pd.DataFrame(image_metadata), pd.DataFrame(pokemon_metadata)

    def __repr__(self) -> str:
        """Return the string representation of the YData object.

        :return: The string representation of the YData object.
        """
        return f"YData({len(self.labels)})"

    def __iter__(self) -> Self:
        """Return the iterator of the YData object.

        :return: The iterator of the YData object.
        """
        return self

    def __next__(self) -> Generator[tuple[npt.NDArray[np.floating[Any]], pd.DataFrame | None, pd.DataFrame | None], None, None]:
        """Get the next labels, image metadata, and Pokémon metadata.

        :return: The next labels, image metadata, and Pokémon metadata.
        """
        for i in range(len(self)):
            yield self[i]
