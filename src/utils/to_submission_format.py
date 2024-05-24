"""Utility function to convert a list of predictions to the submission format."""

import os

import numpy as np
import numpy.typing as npt
import pandas as pd


def to_submission_format(predictions: npt.NDArray[np.float32], test_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Convert a list of predictions to the submission format.

    :param predictions: List of predictions
    :param test_path: Path to the test data

    :return: Submission dataframe
    """
    all_types = [
        "Bug",
        "Dark",
        "Dragon",
        "Electric",
        "Fairy",
        "Fighting",
        "Fire",
        "Flying",
        "Ghost",
        "Grass",
        "Ground",
        "Ice",
        "Normal",
        "Poison",
        "Psychic",
        "Rock",
        "Steel",
        "Water",
    ]
    ids = range(len(os.listdir(test_path)))
    submission = pd.DataFrame(predictions, columns=all_types)

    # Insert id column at the front
    submission.insert(0, "id", ids)

    return submission


# if __name__ == "__main__":
#     predictions = np.random.rand(5000, 18)
#     test_path = "../../data/raw/test_images"
#     submission = to_submission_format(predictions, test_path)
#     print(submission.head())
