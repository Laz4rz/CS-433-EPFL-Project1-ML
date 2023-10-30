#!/usr/bin/env python
# encoding: utf-8
"""
implementations.py: File containing the implementations of the following ML algorithms:
    - Gradient descent
    - Stochastic gradient descent
    - Least squares
    - Ridge regression
    - Logistic regression
    - Regularized logistic regression 
"""

import numpy as np
import src.utils.functions as utils


def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y shape=(N, ):
        tx shape=(N,2):
        w: shape=(2, ). The vector of model parameters.

    Returns:
        g: array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(tx)


def sigmoid(t):
    """Vectorized sigmoid function to improve numerical precision.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """

    return 1.0 / (1 + np.exp(-t))


def compute_loss_logistic(y, tx, w):
    """Compute the cost by negative log likelihood.

    Args:
        y: outpus/labels
        tx: standardized inputs/features augmented with the first column filled with 1's
        w: weights used to calculate loss

    Returns:
        logistic loss
    """

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    y = y.reshape((-1, 1))
    loss = np.sum(np.logaddexp(0, tx.dot(w))) - y.T.dot(tx.dot(w))
    return np.squeeze(loss) * (1 / y.shape[0])


def compute_gradient_logistic(y, tx, w):
    """Compute the gradient of loss for logistic regression.

    Args:
        y: outpus/labels
        tx: standardized inputs/features augmented with the first column filled with 1's
        w: weights

    Returns:
        :return: logistic gradient
    """

    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y) * (1 / y.shape[0])


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm (GD).

    Args:
        y: outpus/labels
        tx: standardized inputs/features augmented with the first column filled with 1's
        initial_w: initial weight vector
        max_iters: number of iterations
        gamma: step size

    Returns:
        loss: loss value of the last iteration of GD
        w: model parameters as numpy arrays of shape of the last iteration of GD
    """
    w = initial_w
    loss = compute_loss(y, tx, w)

    for i in range(max_iters):

        if i % 10 == 0:
            print(f"iteration={i}, loss={loss}")
        g = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * g
        # compute loss, gradient
        loss = compute_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD`
        loss: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    """
    w = initial_w
    batch_size = 1
    n_batches = tx.shape[0] // batch_size
    loss = compute_loss(y, tx, w)

    for i in range(max_iters):

        if i % 10 == 0:
            print(f"iteration={i}, loss={loss}")
        grad = 0
        for batch_y, batch_tx in utils.batch_iter(y, tx, batch_size, n_batches):
            grad += compute_gradient(batch_y, batch_tx, w)
        grad = grad / n_batches
        w = w - gamma * grad
        loss = compute_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Least squares.

    Args:
        y: outpus/labels
        tx: standardized inputs/features augmented with the first column filled with 1's
        lambda_: penalty factor

    Returns:
        loss: loss value of the last iteration
        w: model parameters as numpy arrays of the last iteration

    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression.

    Args:
        y: outpus/labels
        tx: standardized inputs/features augmented with the first column filled with 1's
        lambda_: penalty factor

    Returns:
        loss: loss value of the last iteration
        w: model parameters as numpy arrays of the last iteration
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with loss minimized using gradient descent

    Args:
        y: outpus/labels
        tx: standardized inputs/features augmented with the first column filled with 1's
        initial_w: initial weight vector
        max_iters: number of iterations
        gamma: step size

    Returns:
        w: minimized weight vector
        loss: corresponding loss
    """
    w = initial_w
    y = np.where(y == -1, 0, y)

    loss = compute_loss_logistic(y, tx, w)

    for i in range(max_iters):

        if i % 10 == 0:
            print(f"iteration={i}, loss={loss}")
        gradient = compute_gradient_logistic(y, tx, w)
        w = w - gamma * gradient
        loss = compute_loss_logistic(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using SGD.

    Args:
        y: outpus/labels
        tx: standardized inputs/features augmented with the first column filled with 1's
        lambda_: penalty factor
        initial_w: initial weight vector
        max_iters: number of iterations
        gamma: step size

    Returns:
        w: minimized weight vector
        loss: corresponding loss
    """
    w = initial_w
    y = np.where(y == -1, 0, y)

    loss = compute_loss_logistic(y, tx, w)

    for i in range(max_iters):

        if i % 10 == 0:
            print(f"iteration={i}, loss={loss}")
        gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        loss = compute_loss_logistic(y, tx, w)

    return w, loss
