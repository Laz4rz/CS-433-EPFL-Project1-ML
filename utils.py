import numpy as np


def standardize(x):
    """Standardize the dataset.

    Args:
        x: the dataset to be standardized.
    Returns:
        nx: the standardized dataset.
    """

    nx = np.c_[np.ones((x.shape[0], 1)), x]

    mean_x = np.mean(nx)
    nx = nx - mean_x
    std_x = np.std(nx)
    nx = nx / std_x

    return nx


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

    def sig_elem(z):
        if z <= 0:
            return np.exp(z) / (np.exp(z) + 1)
        else:
            return 1 / (1 + np.exp(-z))

    return np.vectorize(sig_elem)(t)


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
