import numpy as np
from utils import *

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
    for _ in range(max_iters):
        
        # compute loss, gradient
        loss = compute_loss(y, tx, w)
        g = compute_gradient(y, tx, w)
        
        # update w by gradient
        w = w - gamma * g

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

def reg_logistic_regression(y, tx, lambda_):
    """Regularized logistic regression using SGD.

        :param y: outpus/labels
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param lambda_: penalty factor
        :return: (w, loss) where w is the last weight vector and loss is the corresponding loss value
    """
    max_iters = 500
    w = np.ones(tx.shape[1])
    for n_iter in range(max_iters):
        gamma = 1/(n_iter+1)
        for y_b, tx_b in batch_iter(y, tx, batch_size=30, num_batches=1):
            gradient = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
            loss = calculate_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
            w = w - gamma * gradient
    return w, loss