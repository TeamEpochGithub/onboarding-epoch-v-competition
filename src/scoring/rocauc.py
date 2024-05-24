"""ROC AUC scorer from Kaggle."""

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing_extensions import Never

from src.scoring.scorer import Scorer

YT = TypeVar("YT", bound=npt.NBitBase)


@dataclass
class ROCAUC(Scorer):
    """OC AUC scorer from Kaggle."""

    def __call__(self, y_true: npt.NDArray[np.floating[YT]], y_pred: npt.NDArray[np.floating[YT]], **kwargs: Never) -> float:
        """Calculate the ROC AUC score.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :param kwargs: [UNUSED] Additional keyword arguments.
        :return: The ROC AUC score.
        """
        # Convert both solution and submission to a dataframe
        solution = pd.DataFrame(y_true)
        submission = pd.DataFrame(y_pred)

        if not pd.api.types.is_numeric_dtype(submission.values):
            bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pd.api.types.is_numeric_dtype(submission[x])}
            raise ValueError(f"Columns {bad_dtypes} have non-numeric dtypes.")

        solution_sums = solution.sum(axis=0)
        scored_columns = list(solution_sums[solution_sums > 0].index.values)
        # Raise an error if scored columns  <= 0
        if len(scored_columns) <= 0:
            raise ValueError("No positive labels in y_true, ROC AUC score is not defined in that case.")

        # Calculate the ROC AUC score
        return roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average="macro")
