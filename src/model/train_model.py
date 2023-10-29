#!/usr/bin/env python
# encoding: utf-8
"""
train_model.py: File containing the code to train the models.
"""

import numpy as np
import src.model.Models as model
import src.evaluation.evaluation as eval
import src.model.predict_model as predict_model


def build_k_indices(num_samples: int, k: int = 2) -> np.ndarray:
    """Build k indices for k-fold.

    Args:
        num_samples (int): number of samples.
        k (int, optional): number of folds. Defaults to 2.

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
    model: model.Models,
    **kwargs,
) -> [float, float, np.ndarray]:
    """Computes the cross validation for the specified model type.

    Args:
        x (np.ndarray): x data.
        y (np.ndarray): known labels.
        k_indices (np.ndarray): indices of the folds.
        k (int): the kth folds to select.
        algorithm (callable): algorithm to be used.
        model (Models): model type.
        kwargs: additional arguments, parameters for the algorithm.

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
    pred = predict_model.model_functions[model.name](
        x_test=x_test,
        w=w,
    )

    accuracy = eval.compute_accuracy(y=y_test, y_pred=pred)
    f1 = eval.compute_f1_score(y=y_test, y_pred=pred)
    return accuracy, f1, w


def run_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    algorithm: callable,
    model: model.Models,
    **kwargs,
) -> [float, float, np.ndarray]:
    """Computes the cross validation for the specified model type.

    Args:
        x (np.ndarray): x data.
        y (np.ndarray): known labels.
        k (int): the number of folds.
        algorithm (callable): algorithm to be used.
        model (Models): model type.
        kwargs: additional arguments, parameters for the algorithm.

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
