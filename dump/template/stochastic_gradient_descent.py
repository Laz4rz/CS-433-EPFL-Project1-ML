# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter
from costs import compute_loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    N = y.shape[0]
    # N_stoch_idx = N // 100
    # N_stoch_idx = 1
    # stoch_idx = np.random.choice(np.arange(N), N_stoch_idx)

    # y_temp = y[stoch_idx]
    # tx_temp = tx[stoch_idx]

    e = y - tx @ w

    grad = -1 / N * tx.T @ e
    return grad


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    n_batches = tx.shape[0] // batch_size
    for n_iter in range(max_iters):
        grad = 0
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, n_batches):
            grad += compute_stoch_gradient(batch_y, batch_tx, w)
        grad = grad / n_batches
        w = w - gamma * grad  # E[grad(L_n(w))] = grad(L(w)) so we approx
        loss = compute_loss(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws
