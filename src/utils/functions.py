#!/usr/bin/env python
# encoding: utf-8
"""
functions.py: File containing utility functions.
"""

import os
import random
import numpy as np
import helpers as hp
import src.utils.constants as c
import src.model.Models as model
import src.model.predict_model as predict_model
import src.features.build_features as bf

def set_random_seed(seed: int = 42) -> None:
    """Set the random seed.

    Args:
        seed (int): random seed.
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

    Returns:
        np.ndarray: weights.
    """
    if how == "zeros":
        return np.zeros((x.shape[1], 1))
    elif how == "ones":
        return np.ones((x.shape[1], 1))
    elif how == "random":
        return np.random.random((x.shape[1], 1)) * 10

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
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
    x_test: np.ndarray,
    w: np.ndarray,
    model: model.Models,
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

    x_test_standardized = bf.build_test_features(
        x_test=x_test, removed_cols=removed_cols
    )
    pred = predict_model.model_functions[model.name](
        x_test=x_test_standardized,
        w=w,
    )
    test_ids = np.arange(pred.shape[0]) + 1
    os.makedirs(c.MODELS_PATH, exist_ok=True)
    hp.create_csv_submission(
        ids=test_ids, y_pred=pred, name=os.path.join(c.MODELS_PATH, filename)
    )
