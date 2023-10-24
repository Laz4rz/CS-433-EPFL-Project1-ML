# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""

from costs import compute_loss
import numpy as np


def compute_gradient_from_def(y, tx, w, delta=0.00000001):
    """Computes the gradient at w from definition:

    grad(w) = [(L(w0 + delta) - L(w0)) / delta, (L(w1 + delta) - L(w1)) / delta]

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    grad = np.zeros(len(w))
    loss = compute_loss(y, tx, w)
    for enum, _ in enumerate(w):
        w_temp = w.copy()
        w_temp[enum] += delta
        loss_delta_wn = compute_loss(y, tx, w_temp)
        grad[enum] = (loss_delta_wn - loss) / delta
    return grad


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]

    e = y - tx @ w
    grad = -1 / N * tx.T @ e
    return grad


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, ws[-1])
        w = ws[-1] - gamma * grad  # E[grad(L_n(w))] = grad(L(w)) so we approx
        loss = compute_loss(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws
