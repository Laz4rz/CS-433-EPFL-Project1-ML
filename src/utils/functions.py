#!/usr/bin/env python
# encoding: utf-8
"""
functions.py: File containing utility functions.
"""

import os
import random
import pickle
import numpy as np
import helpers as hp
from typing import Tuple
import src.utils.constants as c
import src.model.Models as model
import src.model.predict_model as predict_model
import src.features.build_features as bf


def get_all_results_values(results: dict, key: str) -> list:
    return list(map(lambda x: x[key], results.values()))


def pickle_results(results: dict, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(results, f)


def unpickle_results(filename: str) -> dict:
    with open(filename, "rb") as f:
        return pickle.load(f)


def set_random_seed(seed: int = 42) -> None:
    """Set the random seed.

    Args:
        seed (int, optional): random seed.
    """
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def initialize_weights(x: np.ndarray, how: str = None) -> np.ndarray:
    """Initialize the weights.

    Args:
        x (np.ndarray): dataset.
        how (str, optional): how to initialize the weights. Defaults to None.

    Returns:
        np.ndarray: weights.
    """
    if how == "zeros":
        return np.zeros((x.shape[1], 1))
    elif how == "ones":
        return np.ones((x.shape[1], 1))
    elif how == "random":
        return np.random.random((x.shape[1], 1))


def batch_iter(
    y: np.ndarray,
    tx: np.ndarray,
    batch_size: int = 1,
    num_batches: int = 1,
    shuffle: bool = True,
):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Args:
        y (np.ndarray): labels.
        tx (np.ndarray): data.
        batch_size (int, optional): size of the batch. Defaults to 1.
        num_batches (int, optional): number of batches. Defaults to 1.
        shuffle (bool, optional): whether to shuffle or not. Defaults to True.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def create_submission(
    x: np.ndarray,
    ids: np.ndarray,
    w: np.ndarray,
    model: model.Models,
    idx_calc_columns: np.ndarray = [],
    idx_nan_percent: np.ndarray = [],
    fill_nans: str = None,
    num: int = 0,
    filename: str = "sub.csv",
    degree: int = 1,
) -> None:
    """Creates the submission file for the specified model type.

    Args:
        x (np.ndarray): test data (to be standardized).
        ids (np.ndarray): indexes of the data.
        w (np.ndarray): weights.
        idx_calc_columns (np.ndarray, optional): indexes of the calculated columns. Defaults to [].
        idx_nan_percent (np.ndarray, optional): indexes of the columns with more than percentage NaN values. Defaults to [].
        fill_nans (str, optional): method to fill NaN values. Defaults to None.
        num (int, optional): value to fill NaN values (if fill_nans == with_num). Defaults to 0.
        file_name (str, optional): file name. Defaults to "sub.csv".
        degree (int, optional): degree of the polynomial expansion. Defaults to 1.
    """
    x_test_standardized = bf.build_test_features(
        x=x,
        idx_calc_columns=idx_calc_columns,
        idx_nan_percent=idx_nan_percent,
        fill_nans=fill_nans,
        num=num,
        polynomial_expansion_degree=degree,
    )
    pred = predict_model.model_functions[model.name](
        x_test=x_test_standardized,
        w=w,
    )
    os.makedirs(c.MODELS_PATH, exist_ok=True)
    hp.create_csv_submission(
        ids=ids, y_pred=pred, name=os.path.join(c.MODELS_PATH, filename)
    )


def get_losses_at_each_iter(
    x: np.ndarray,
    y: np.ndarray,
    algorithm: callable,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulates the losses and weights at each iteration.

    Args:
        x (np.ndarray): dataset.
        y (np.ndarray): labels.
        algorithm (callable): algorithm to use.

    Returns:
        np.ndarray: losses.
        np.ndarray: weights.
    """

    iter = kwargs["max_iters"]
    kwargs["max_iters"] = 1

    losses = []
    weights = []

    for _ in range(iter):
        w, loss = algorithm(y=y, tx=x, **kwargs)
        losses.append(loss)
        weights.append(w)
        kwargs["initial_w"] = w

    return np.array(losses), np.array(weights)
