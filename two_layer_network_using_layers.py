# coding: utf-8
import os
import sys
import numpy as np
from functions import numerical_gradient
from layers import Relu, Affine, SoftmaxWithLoss, Sigmoid
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
        # Create layer
        self.layers = {}    # if it is used ~ Python 3.6: OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        # self.layers["Sigmo1"] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """Predict which value is likely

        Args:
            x (numpy.ndarray): image data which mean input to NN

        Returns:
            [numpy.ndarray]: predict probability
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """Loss function using cross entropy error

        Args:
            x (numpy.ndarray): image data which mean input to NN
            t (numpy.ndarray): labels

        Returns:
            float: result of cross entropy error
        """
        y = self.predict(x)
        loss = self.lastLayer.forward(y, t)
        return loss

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

        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
