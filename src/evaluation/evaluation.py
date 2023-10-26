#!/usr/bin/env python
# encoding: utf-8
"""
evaluation.py: File containing the code to evaluate the models.
"""

import numpy as np


def compute_rmse(loss: float) -> float:
    """Compute the rmse.

    Args:
        loss (float): loss.

    Returns:
        float: rmse.
    """
    return np.sqrt(2 * loss)


def compute_accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the accuracy.

    Args:
        y (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.

    Returns:
        float: accuracy.
    """
    return np.sum(y == y_pred) / len(y)


def compute_f1_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the f1 score.

    Args:
        y (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.
    Returns:
        float: f1 score.
    """
    true_positives = np.sum(y[y == 1] == y_pred[y_pred == 1])
    false_positives = np.sum(y[y == 1] != y_pred[y_pred == 1])
    false_negatives = np.sum(y[y == -1] != y_pred[y_pred == -1])
    return true_positives / (true_positives + (false_positives + false_negatives) / 2)
