import numpy as np

def compute_loss(y, tx, w, loss_type='mse'):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx.dot(w)

    # MSE loss
    if loss_type == 'mse':    
        loss = 1/(2*len(y)) * e.T.dot(e)
        
    # MAE loss
    elif loss_type == 'mae':
        loss = 1/(2*len(y)) * np.sum(np.abs(e))
    
    else:
        raise ValueError("loss_type must be either 'mse' or 'mae'")
    
    return loss

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y shape=(N, ): 
        tx shape=(N,2): 
        w: shape=(2, ). The vector of model parameters.

    Returns:
        g: array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************

    e = y - tx.dot(w)
    g  = -1/len(y) * tx.T.dot(e)
    
    return g