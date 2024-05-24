"""Test for the add_constant.py module."""

import numpy as np

from src.data.xdata import XData
from src.modules.transformation.x.add_constant import AddConstant


def test_add_constant() -> None:
    """Test the add_constant.py module."""
    images = XData(np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]), None, None)

    add_constant = AddConstant(5)
    results = add_constant.transform(images)

    assert np.allclose(results.images, np.array([
        [5.1, 5.2, 5.3],
        [5.4, 5.5, 5.6],
        [5.7, 5.8, 5.9]
    ]))
