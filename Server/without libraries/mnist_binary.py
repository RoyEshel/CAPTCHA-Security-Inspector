import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from layers import Dense, Convolutional, Flatten
from activations import Sigmoid
from losses import BinaryCrossEntropy as BCE
from network import train, predict


def preprocess_data(x, y, limit):
    zero_indexes = np.where(y == 0)[0][:limit]
    one_indexes = np.where(y == 1)[0][:limit]
    binary_indexes = np.hstack((zero_indexes, one_indexes))
    binary_indexes = np.random.permutation(binary_indexes)
    x, y = x[binary_indexes], y[binary_indexes]
    x = x.reshape(len(x), 1, 28, 28) / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 3, 5),
    Sigmoid(),
    Flatten((5, 26, 26)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

# train
train(network, x_train, y_train, BCE.binary_cross_entropy, BCE.binary_cross_entropy_prime, epochs=20, learning_rate=0.1)

# test
success_counter=0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    success_counter += (np.argmax(output) == np.argmax(y))

print(f'success rate: {round(success_counter / len(x_test) * 100, 2)}%')

