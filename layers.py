# coding: utf-8
import numpy as np
from functions import sigmoid, softmax, cross_entropy_error
# from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """forword propagation

        Args:
            x (numpy.ndarray): input

        Returns:
            numpy.ndarray: output
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """backward propagation

        Args:
            dout (numpy.ndarray): propagated dout from right layer

        Returns:
            numpy.ndarray: dout value
        """
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        """forword propagation

        Args:
            x (numpy.ndarray): input

        Returns:
            numpy.ndarray: output
        """
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        """backward propagation

        Args:
            dout (numpy.ndarray): propagated dout from right layer

        Returns:
            numpy.ndarray: dout value
        """
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W  # wight
        self.b = b  # bias

        self.x = None  # input
        self.dW = None  # dout of wight
        self.db = None  # dout of bias
        # self.original_x_shape = None

    def forward(self, x):
        """forward propagation

        Args:
            x (numpy.ndarray): input

        Returns:
            numpy.ndarray: output
        """
        # テンソル対応
        # self.original_x_shape = x.shape
        # x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        """backward propagation

        Args:
            dout (numpy.ndarray): propagated dout from right layer

        Returns:
            [type]: dout value
        """
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # output of softmax
        self.t = None  # train data

    def forward(self, x, t):
        """forward propagation

        Args:
            x (numpy.ndarray): input
            t (numpy.ndarray): train data

        Returns:
            float: cross entropy error
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """backward propagation

        Args:
            dout (float, optional): dout from right layer. Defaults to 1.

        Returns:
            numpy.ndarray: dout value
        """
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = (self.y - self.t) * (dout / batch_size)

        return dx
