from two_layer_network import TwoLayerNet
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.pardir)

# Load train data and test data
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

# Create two layer NN
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# Set params
iters_num = 10000  # how many loop
batch_size = 100  # batch size
learning_rate = 0.1  # learning rate

# Record list of results
train_loss_list = []  # loss function
train_acc_list = []  # accuracy of train data
test_acc_list = []  # accuracy of test data

train_size = x_train.shape[0]  # train data size
iter_per_epoch = max(train_size / batch_size, 1)  # how many loop per epoch

# Start learning
for i in range(iters_num):
    # Get mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Calculate grad
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # Update params
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Caluculate loss function
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # Caluculate accuracy per one epoch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# plot accuracy of train data and test data
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# plot transition the value of the loss function
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='loss')
plt.xlabel("iteration")
plt.ylabel("loss")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()
