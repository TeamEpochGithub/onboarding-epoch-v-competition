"""Tests for the one_hot.py module."""
import numpy as np
import pandas as pd

from src.data.ydata import YData
from src.modules.transformation.y.one_hot import OneHot


def test_one_hot() -> None:
    """Test the one_hot.py module."""

    labels = YData(np.array([
        [0.1, 0.7, 0.3],
        [0.1, 0.5, 0.6],
        [0.8, 0.4, 0.1],
    ]), None, None)

    one_hot = OneHot()
    results = one_hot.transform(labels)

    assert np.array_equal(results.labels, np.array([
        [False, True, False],
        [False, False, True],
        [True, False, False],
    ]))
