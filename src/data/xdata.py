"""Dataclass for the X data."""

from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import Self

from src.data import IlocIndexer


@dataclass
class XData:
    """Dataclass for the X data.

    :param images: The images.
    :param image_metadata: The image metadata.
    :param pokemon_metadata: The Pokémon metadata.
    """

    images: Sequence[npt.NDArray[np.floating[Any]]] | npt.NDArray[np.floating[Any]]  # List of images with length N and shape (Height, Width) / (Height, Width, Channels)
    image_metadata: pd.DataFrame | None = None  # Shape (N, 5)
    pokemon_metadata: pd.DataFrame | None = None  # Shape (N, 15)

    def __bool__(self) -> bool:
        """Return whether the XData object has images.

        :return: Whether the XData object has images.
        """
        return len(self.images) > 0

    def __len__(self) -> int:
        """Return the number of images in the XData object.

        :return: The number of images in the XData object.
        """
        return len(self.images)

    def __getitem__(self, key: IlocIndexer) -> tuple[npt.NDArray[np.floating[Any]], pd.DataFrame | None, pd.DataFrame | None]:
        """Get the images, image metadata, and Pokémon metadata at the specified index or indices.

        :param key: The index or indices to get the images, image metadata, and Pokémon metadata.
        :return: The images, image metadata, and Pokémon metadata at the specified index or indices.
        """
        try:
            images = self.images[key]  # type: ignore[index]
        except TypeError as e:
            msg = "Did you make sure that your images in XData is a numpy array before feeding it into the model? Tip: Also take a look at handling the shape of the images."
            raise TypeError(msg) from e

        if self.image_metadata is None or self.pokemon_metadata is None:
            return images, None, None

        image_metadata = self.image_metadata.iloc[key]  # type: ignore[index]
        pokemon_metadata = self.pokemon_metadata[self.pokemon_metadata["Pokemon"].isin(image_metadata["Pokemon"])]

        return images, pd.DataFrame(image_metadata), pd.DataFrame(pokemon_metadata)

    def __repr__(self) -> str:
        """Return the string representation of the XData object.

        :return: The string representation of the XData object.
        """
        return f"XData({len(self.images)})"

    def __iter__(self) -> Self:
        """Return the iterator of the XData object.

        :return: The iterator of the XData object.
        """
        return self

    def __next__(self) -> Generator[tuple[npt.NDArray[np.floating[Any]], pd.DataFrame | None, pd.DataFrame | None], None, None]:
        """Return the next images, image metadata, and Pokémon metadata.

        :return: The next images, image metadata, and Pokémon metadata.
        """
        for i in range(len(self)):
            yield self[i]
