import random
import numpy as np
from keras.datasets import mnist

from layers import Dense, Convolutional, Flatten
from activations import Sigmoid, Softmax, ReLU
from losses import SparseCategoricalCrossEntropy as SCCE
from network import train, predict


def preprocess_data(x, y, limit):
    indexes = [i for i in range(10000)]
    for i in range(10):
        random.shuffle(indexes)
    indexes = indexes[: limit]

    x, y = x[indexes], y[indexes]
    x = x.reshape(len(x), 1, 28, 28) / 255
    return x, y


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 3200)  # 50 goes for validation
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 3, 5),
    ReLU(),
    Flatten((5, 26, 26)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Softmax(),
]

# train
train(network, x_train, y_train, SCCE.sparse_categorical_cross_entropy, SCCE.sparse_categorical_cross_entropy_prime, val_split=50/3200, epochs=200, learning_rate=0.01)

# test
success_counter = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {y}")
    success_counter += (np.argmax(output) == y)

print(f'success rate: {round(success_counter / len(x_test) * 100, 2)}%')

