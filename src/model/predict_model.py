#!/usr/bin/env python
# encoding: utf-8
"""
predict_model.py: File containing the code to compute the predictions.
"""

import numpy as np
import implementations as impl
import src.model.Models as model


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


def compute_predictions_logistic(x_test: np.ndarray, w: np.ndarray, threshold: int = 0.2) -> np.ndarray:
    """Computes the predictions using the logistic regression model.

    Args:
        x_test (np.ndarray): test data
        w (np.ndarray): weights

    Returns:
        y_pred (np.ndarray): predictions
    """
    y_pred = np.dot(x_test, w)
    y_pred = impl.sigmoid(t=y_pred)
    y_pred = np.where(y_pred >= 0.5, 1, -1)
    return y_pred


model_functions = {
    model.Models.LINEAR.name: compute_predictions_linear,
    model.Models.LOGISTIC.name: compute_predictions_logistic,
}
