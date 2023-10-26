#!/usr/bin/env python
# encoding: utf-8
"""
predictions.py: File containing the code to compute the predictions.
"""

import os
import numpy as np
import src.constants as c
from enum import Enum
from src.utils import sigmoid
from src.data.build_data import build_test_data
from helpers import create_csv_submission


def compute_predictions_linear(x_test: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the predictions using the linear regression model.

    Args:
        x_test (np.ndarray): test data.
        w (np.ndarray): weights.

    Returns:
        y_pred (np.ndarray): predictions.
    """

    y_pred = np.dot(x_test, w)
    y_pred = np.where(y_pred >= 0, 1, -1)
    return y_pred


def compute_predictions_logistic(x_test: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the predictions using the logistic regression model.

    Args:
        x_test (np.ndarray): test data
        w (np.ndarray): weights

    Returns:
        y_pred (np.ndarray): predictions
    """

    y_pred = np.dot(x_test, w)
    y_pred = sigmoid(t=y_pred)
    y_pred = np.where(y_pred >= 0.5, 1, -1)
    return y_pred


class Models(Enum):
    LINEAR = "linear"
    LOGISTIC = "logistic"


model_functions = {
    Models.LINEAR: compute_predictions_linear,
    Models.LOGISTIC: compute_predictions_logistic,
}


def create_submission(
    x_test: np.ndarray,
    w: np.ndarray,
    model: Models,
    removed_cols: np.ndarray = [],
    filename: str = "sub.csv",
) -> None:
    """Creates the submission file for the specified model type.

    Args:
        x_test (np.ndarray): test data (to be standardized).
        w (np.ndarray): weights.
        removed_cols (np.ndarray): indexes of the columns removed from the training data.
        function (callable): function used to compute the predictions.
        file_name (str): file name.
    """

    x_test_standardized = build_test_data(x_test=x_test, removed_cols=removed_cols)
    pred = model_functions[model](
        x_test=x_test_standardized,
        w=w,
    )
    test_ids = np.arange(pred.shape[0]) + 1
    os.makedirs(c.MODELS_PATH, exist_ok=True)
    create_csv_submission(
        ids=test_ids, y_pred=pred, name=os.path.join(c.MODELS_PATH, filename)
    )
