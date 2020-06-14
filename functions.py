# coding: utf-8
import os
import sys
import numpy as np

sys.path.append(os.pardir)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    """Sigmoid's grad Function which we use in backwardpropagation. """
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    """Softmax Function
        Function that converts and outputs so that the total
        of multiple output values is 1.0 (= 100%)
    """
    x = x - np.max(x, axis=-1, keepdims=True)   # avoid overflow
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def mean_squared_error(y, t):
    """Loss Function(using mean squared error)

    Args:
        y (numpy.ndarray): output of neural network
                           such as [0.1, 0.05, 0.6, 0.0, 0.05 ...]
        t ([numpy.ndarray): labels such as [0, 0, 1, 0, 0 ...]

    Returns:
        float: a value of mean squared error
    """
    return 0.5 * sum((y-t)**2)


def cross_entropy_error(y, t):
    """Loss Function( using cross entropy error)

    Args:
        y (numpy.ndarray): output of neural network
        t (numpy.ndarray): labels

    Returns:
        float: a value of cross entropy error
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # When training data is one-hot-vector, conversion to label's index
    # one-hot-vector means just one element is one and the onters are zero
    # such as [0,0,1,0,0,0]
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_gradient(f, x):
    h = 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val
        it.iternext()

    return grad
