"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""

from os import PathLike
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from src.data.xdata import XData
from src.data.ydata import YData


def setup_train_x_data(path: str | PathLike[str], image_metadata: str | PathLike[str], pokemon_metadata: str | PathLike[str]) -> XData:
    """Create train S data for pipeline.

    This is already implemented for you. Feel free to modify it if you need to.

    :param path: The path to the training images.
    :param image_metadata: Metadata for the images.
    :param pokemon_metadata: Metadata for the Pokémon.
    :return: x data
    """
    # Load the metadata
    image_metadata = pd.read_csv(image_metadata)
    valid_metadata = image_metadata[image_metadata['Type'] != 'Back']
    pokemon_metadata = pd.read_csv(pokemon_metadata)

    images: list[npt.NDArray[np.floating[Any]]] = []

    # Load the images from the full path using imageio
    for i in tqdm(image_metadata["Full_path"], desc="Loading images..."):
        load_path = path + i
        image = iio.imread(load_path)
        images.append(image)

    return XData(images=images, image_metadata=valid_metadata.reset_index(drop=True), pokemon_metadata=pokemon_metadata)


def setup_train_y_data(image_metadata: str | PathLike[str], pokemon_metadata: str | PathLike[str]) -> YData:
    """Create train y data for pipeline.

    This is already implemented for you. Feel free to modify it if you need to.

    :param image_metadata: Metadata for the images
    :param pokemon_metadata: Metadata for the Pokémon
    :return: y data
    """
    # Load the metadata
    image_metadata = pd.read_csv(image_metadata)
    valid_metadata = image_metadata[image_metadata['Type'] != 'Back']
    pokemon_metadata = pd.read_csv(pokemon_metadata)

    # Merge the dataframes to find the types of the training images
    test_labels = image_metadata.merge(pokemon_metadata, left_on="Pokemon", right_on="Pokemon", how="left")

    # Get the types
    all_labels = test_labels["Type_y"]

    # Get all possible types and split by ,
    all_types = set()
    for label in all_labels:
        all_types.update(label.split(", "))

    # Convert to list and alphabetically sort
    all_types = list(all_types)
    all_types.sort()

    # Create a dictionary with the type as key and the index as value
    type_dict = {all_types[i]: i for i in range(len(all_types))}

    # Convert the test_labels to the type index
    test_labels["Type_index"] = test_labels["Type_y"].apply(lambda x: [type_dict[y] for y in x.split(", ")])

    one_hot_labels = np.zeros((len(test_labels), len(all_types)))

    for i, row in test_labels.iterrows():
        for type_index in row["Type_index"]:
            one_hot_labels[i, type_index] = 1

    return YData(labels=one_hot_labels, image_metadata=valid_metadata.reset_index(drop=True), pokemon_metadata=pokemon_metadata)


def setup_inference_data(path: str | PathLike[str]) -> XData:
    """Create data for inference with pipeline.

    This function is already implemented for you.

    :param path: Path to the test images.
    :return: Inference data
    """
    path: Path = Path(path)
    filenames = [f for f in path.glob("**/*.png") if f.is_file()]

    images: list[npt.NDArray[np.floating[Any]]] = []

    # Loads all images
    for i in tqdm(range(len(filenames)), desc="Loading test images..."):
        image = iio.imread(path / f"{i}.png")
        images.append(image)

    return XData(images=images, image_metadata=None, pokemon_metadata=None)


def setup_splitter_data() -> None:
    """Create data for splitter."""
    return
