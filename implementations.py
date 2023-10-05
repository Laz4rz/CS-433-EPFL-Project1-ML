import numpy as np
from utils import *

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm (GD).

    Args:
        y ( shape=(N, ) ): 
        tx ( shape=(N,2) ): 
        initial_w ( shape=(2, ) ): initial weight vector
        maxORACLE invites you to join them for an exclusive insight into Oracle Labs and for an overview of their internship opportunities on Wednesday, 25 October 2023 à 5.15 p.m. in Room BC410 ._iters ( int ): number of steps to run
        gamma ( int ): step size
    
    Returns:
        loss: loss value (scalar) of the last iteration of GD
        w: model parameters as numpy arrays of shape (2, ) of the last iteration of GD
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
    """calculate the least squares."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

#def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma);
