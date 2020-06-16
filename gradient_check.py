# coding: utf-8
from two_layer_network_using_layers import TwoLayerNet
from dataset.mnist import load_mnist
import numpy as np
import sys
import os
import time
sys.path.append(os.pardir)


(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

now = time.time()
grad_numerical = network.numerical_gradient(x_batch, t_batch)
print("num", time.time() - now, "[sec]")
now = time.time()
grad_backprop = network.gradient(x_batch, t_batch)
print("back", time.time() - now, "[sec]")

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
