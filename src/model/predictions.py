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
    """Enumeration of the models.

    Args:
        Enum (str): enumeration of the models.
    """

    LINEAR = "linear"
    LOGISTIC = "logistic"


model_functions = {
    Models.LINEAR.name: compute_predictions_linear,
    Models.LOGISTIC.name: compute_predictions_logistic,
}


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


def build_k_indices(num_samples: int, k: int) -> np.ndarray:
    """Build k indices for k-fold.

    Args:
        num_samples (int): number of samples.
        k (int): number of folds.

    Returns:
        np.ndarray: Array containing the indices of the folds.
    """

    interval = int(num_samples / k)
    np.random.seed()
    indices = np.random.permutation(num_samples)
    k_indices = [indices[ki * interval : (ki + 1) * interval] for ki in range(k)]
    return np.array(k_indices)


def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    k_indices: np.ndarray,
    kth: int,
    algorithm: callable,
    model: Models,
    **kwargs,
) -> [float, float, float, np.ndarray]:
    """Computes the cross validation for the specified model type.

    Args:
        x (np.ndarray): x data.
        y (np.ndarray): known labels.
        k_indices (np.ndarray): indices of the folds.
        k (int): the kth folds to select.
        algorithm (callable): algorithm to be used.
        model (Models): model type.

    Returns:
        float: accuracy.
        float: f1 score.
        np.ndarray: weights.
    """

    te_indice = k_indices[kth]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == kth)]
    tr_indice = tr_indice.reshape(-1)
    x_train = x[tr_indice]
    y_train = y[tr_indice]
    x_test = x[te_indice]
    y_test = y[te_indice]

    w, _ = algorithm(y=y_train, tx=x_train, **kwargs)
    pred = model_functions[model.name](
        x_test=x_test,
        w=w,
    )

    accuracy = compute_accuracy(y=y_test, y_pred=pred)
    f1 = compute_f1_score(y=y_test, y_pred=pred)
    return accuracy, f1, w


def run_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    algorithm: callable,
    model: Models,
    **kwargs,
) -> [float, float, float, np.ndarray]:
    """Computes the cross validation for the specified model type.

    Args:
        x (np.ndarray): x data.
        y (np.ndarray): known labels.
        k (int): the number of folds.
        algorithm (callable): algorithm to be used.
        model (Models): model type.

    Returns:
        float: accuracy.
        float: f1 score.
        np.ndarray: weights.
    """

    k_indices = build_k_indices(num_samples=x.shape[0], k=k)
    accuracies = []
    f1s = []
    weights = []
    for kth in range(k):
        accuracy, f1, w = cross_validation(
            x=x,
            y=y,
            k_indices=k_indices,
            kth=kth,
            algorithm=algorithm,
            model=model,
            **kwargs,
        )

        accuracies.append(accuracy)
        f1s.append(f1)
        weights.append(w)

    return np.mean(accuracies), np.mean(f1s), np.mean(weights, axis=0)


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
    pred = model_functions[model.name](
        x_test=x_test_standardized,
        w=w,
    )
    test_ids = np.arange(pred.shape[0]) + 1
    os.makedirs(c.MODELS_PATH, exist_ok=True)
    create_csv_submission(
        ids=test_ids, y_pred=pred, name=os.path.join(c.MODELS_PATH, filename)
    )
