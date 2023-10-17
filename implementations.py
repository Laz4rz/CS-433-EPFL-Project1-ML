import numpy as np
from utils import *

def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
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
    n_batches = tx.shape[0] // batch_size

    for _ in range(max_iters):
        grad = 0
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, n_batches):
            grad += compute_gradient(batch_y, batch_tx, w)
        grad = grad / n_batches
        w = w - gamma * grad 
        loss = compute_loss(y, tx, w)
        
    return w, loss

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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with loss minimized using gradient descent

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
    pass

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
    for _ in range(max_iters):
        # gamma = 1/(n_iter+1) not sure? Gamma is passed as constant in the project description
        for y_b, tx_b in batch_iter(y, tx, batch_size=30, num_batches=1):
            gradient = compute_gradient_logistic(y_b, tx_b, w) + 2 * lambda_ * w
            loss = compute_loss_logistic(y_b, tx_b, w) + lambda_ * np.squeeze(w.T.dot(w))
            w -= gamma * gradient
    return w, loss
