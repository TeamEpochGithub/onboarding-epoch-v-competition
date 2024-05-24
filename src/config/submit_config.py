"""Schema for the submit configuration."""

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass
class SubmitConfig(DictConfig):
    """Schema for the submit configuration.

    :param model: Model pipeline.
    :param ensemble: Ensemble pipeline.
    :param post_ensemble: Post ensemble pipeline.
    :param test_path: Path to the test images.
    :param cache_path: Path to the raw target.
    :param processed_path: Path to put processed data.
    :param result_path: Path to the result.
    """

    model: Any
    ensemble: Any
    post_ensemble: Any
    test_path: str
    cache_path: str
    processed_path: str
    result_path: str
