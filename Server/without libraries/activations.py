from layers import Layer
import numpy as np


# Base Activation
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Hadamard Product (⊙ or *) is element wise multiplication
        input_gradient = output_gradient * self.activation_prime(self.input)  # dE/dX = dE/dY ⊙ dY/dX
        return input_gradient


# Hyperbolic Tangent (tanh) activation function and its derivative
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)  # f(x) = tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2  # f'(x) = 1 - tanh(x)^2
        super().__init__(tanh, tanh_prime)


# Sigmoid activation function and its derivative
class Sigmoid(Activation):
    def __init__(self):
        # f(x) = 1 / (1 + e^(-x))
        def sigmoid(x):
            # Clip x to a reasonable range
            x = np.clip(x, -700, 700)
            return 1 / (1 + np.exp(-x))

        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))  # f'(x) = f(x) * (1 - f(x))
        super().__init__(sigmoid, sigmoid_prime)


# Rectified Linear Unit (ReLU) activation function and its derivative
class ReLU(Activation):
    def __init__(self):
        relu = lambda x: x * (x > 0).astype(int)
        relu_prime = lambda x: (x > 0).astype(int)
        super().__init__(relu, relu_prime)


# Softmax activation forward and backward (unlike the others, it can't use the super's forward and backward)
class Softmax(Layer):
    def forward(self, input):
        self.input = input
        # normalize the input to be between -∞ and 0 instead of between -∞ and ∞
        normalized_input = self.input - np.max(self.input)
        tmp = np.exp(normalized_input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        input_gradient = np.dot(self.output * (np.identity(n) - self.output.T), output_gradient)
        return input_gradient
