"""Module for a verbose training block that logs to the terminal and to W&B."""

from epochalyst.pipeline.model.training.training_block import TrainingBlock

from src.modules.logging.logger import Logger


class VerboseTrainingBlock(TrainingBlock, Logger):
    """A verbose training block that logs to the terminal and to W&B.

    To use this block, inherit and implement the following methods:
    - custom_train(x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]
    - custom_predict(x: Any, **pred_args: Any) -> Any
    """
