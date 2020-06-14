# coding: utf-8
import os
import sys
import numpy as np
from functions import sigmoid, softmax, cross_entropy_error,\
    numerical_gradient, sigmoid_grad
sys.path.append(os.pardir)


class TwoLayerNet:

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_std: float = 0.01):
        # init weight
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """Predict which value is likely

        Args:
            x (numpy.ndarray): image data which mean input to NN

        Returns:
            [numpy.ndarray]: predict probability using softmax function
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        #Calculate NN (forward)
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """Loss function using cross entropy error

        Args:
            x (numpy.ndarray): image data which mean input to NN
            t (numpy.ndarray): labels

        Returns:
            float: result of cross entropy error
        """
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """Accuracy

        Args:
            x (numpy.ndarray): image data which mean input to NN
            t (numpy.ndarray): labels

        Returns:
            float: comparison labels with output of neural network
                   and return correct probability
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """Calculate gradient to weight params using numerical gradient.

        Args:
            x (numpy.ndarray): image data which mean input to NN
            t (numpy.ndarray): labels

        Return:
            dictionary: dictionary of gradient to each param.
        """
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """Calculate gradient to weight params using backpropagation.
           This can calculate faster than numerical gradient function.

        Args:
            x (numpy.ndarray): image data which mean input to NN
            t (numpy.ndarray): labels

        Return:
            dictionary: dictionary of gradient to each param.
        """

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
