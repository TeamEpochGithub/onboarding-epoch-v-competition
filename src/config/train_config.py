"""Schema for the train configuration."""

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from src.config.wandb_config import WandBConfig


@dataclass
class TrainConfig(DictConfig):
    """Schema for the train configuration.

    :param model: The model pipeline.
    :param ensemble: The ensemble pipeline.
    :param train_path: Path to the training images.
    :param cache_path: Path to the raw target.
    :param processed_path: Path to put processed data.
    :param scorer: Scorer object to be instantiated.
    :param wandb: Whether to log to Weights & Biases and other settings.
    :param splitter: Cross validation splitter.
    :param test_size: Size of the test set.
    :param allow_multiple_instances: Whether to allow multiple instances of training at the same time.
    """

    model: Any  # ModelPipeline
    ensemble: Any  # EnsemblePipeline
    train_path: str
    pokemon_metadata_path: str
    image_metadata_path: str
    cache_path: str
    processed_path: str
    scorer: Any  # src.scorer.Scorer
    wandb: WandBConfig
    splitter: Any
    test_size: float = 0.2
    allow_multiple_instances: bool = False
