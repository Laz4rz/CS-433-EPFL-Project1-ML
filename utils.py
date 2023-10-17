import numpy as np

def standardize(x):
    """
        The value -999 is used as a placeholder for missing values (undefined).
        There is an interesting value "PRI_JET_NUM":.
    """
    
    x = np.c_[np.ones((x.shape[0], 1)), x]
    
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    
    return x

def compute_loss(y, tx, w, loss_type='mse'):
    """Calculate the loss using either MSE, MAE or logistic regression cost function.

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

def compute_logistic_regression_loss(y, tx, w): 
    """
    """
    def theta(x: np.ndarray) -> np.ndarray:
        return 1 / (1+np.exp(-x))
      
    return 1/len(y) * ((y.T @ np.log(theta(tx @ w))) + (1 - y).T @ np.log(theta(tx @ w)))

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
    g  = -1/len(y) * tx.T.dot(e)
    
    return g

def sigmoid(t):
    """Apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    
    return 1.0 / (1 + np.exp(-t))
    
    # return np.where(t >= 0, 
    #                 1 / (1 + np.exp(-t)), 
    #                 np.exp(t) / (1 + np.exp(t)))

def calculate_loss_logistic(y, tx, w):
    """Compute the cost by negative log likelihood.

    Args:
        y: outpus/labels
        tx: standardized inputs/features augmented with the first column filled with 1's
        w: weights used to calculate loss
    
    Returns:
        logistic loss
    """
    
    y = y.reshape((-1, 1))
    return np.sum(np.logaddexp(0, tx.dot(w))) - y.T.dot(tx.dot(w))

def calculate_gradient_logistic(y, tx, w):
    """Compute the gradient of loss.

    Args:
        y: outpus/labels
        tx: standardized inputs/features augmented with the first column filled with 1's
        w: weights
        
    Returns:
        :return: logistic gradient
    """
    
    return tx.T.dot(sigmoid(tx.dot(w))-y)

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