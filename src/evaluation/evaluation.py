#!/usr/bin/env python
# encoding: utf-8
"""
evaluation.py: File containing the code to evaluate the models.
"""

import numpy as np

from typing import Callable, Dict
from src.utils.parameters import Parameters


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
    true_positives = np.sum(np.logical_and(y == 1, y_pred == 1))
    false_positives = np.sum(np.logical_and(y == -1, y_pred == 1))
    false_negatives = np.sum(np.logical_and(y == 1, y_pred == -1))
    return true_positives / (true_positives + (false_positives + false_negatives) / 2)


def full_evaluation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    w: np.ndarray,
    results: np.ndarray,
    parameters: Parameters,
    compute_predictions_func: Callable,
    loss: float,
) -> Dict:
    """Compute the full evaluation of the model.

    Args:
        x_train (np.ndarray): training data.
        y_train (np.ndarray): training labels.
        x_train_full (np.ndarray): full training data.
        y_train_full (np.ndarray): full training labels.
        x_test (np.ndarray): test data.
        y_test (np.ndarray): test labels.
        w (np.ndarray): weights.
        results (np.ndarray): results.
        parameters (Parameters): parameters.
        compute_predictions_func (Callable): function to compute the predictions.
        loss (float): loss.
    Returns:
        Dict: results."""
    print("\nTraining set:")
    f1_training = compute_f1_score(y_train, compute_predictions_func(x_train, w))
    acc_training = compute_accuracy(y_train, compute_predictions_func(x_train, w))
    print(f"F1 score on training set: {f1_training}")
    print(f"Accuracy on training set: {acc_training}")

    print("\nTest set:")
    f1_test = compute_f1_score(y_test, compute_predictions_func(x_test, w))
    acc_test = compute_accuracy(y_test, compute_predictions_func(x_test, w))
    print(f"F1 score on test set: {f1_test}")
    print(f"Accuracy on test set: {acc_test}")

    print("\nFull set:")
    f1_full = compute_f1_score(y_train_full, compute_predictions_func(x_train_full, w))
    acc_full = compute_accuracy(y_train_full, compute_predictions_func(x_train_full, w))
    print(f"F1 score on full set: {f1_full}")
    print(f"Accuracy on full set: {acc_full}")

    print(f"\nLoss on training set: {loss}")

    print("=" * 50)

    results[str(parameters)] = {
        "f1_training": f1_training,
        "acc_training": acc_training,
        "f1_test": f1_test,
        "acc_test": acc_test,
        "f1_full": f1_full,
        "acc_full": acc_full,
        "loss": loss,
    }
    return results
