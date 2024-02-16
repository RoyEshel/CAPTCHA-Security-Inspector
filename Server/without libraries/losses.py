import numpy as np


# Mean Squared Error (MSE) loss function and its derivative
class MeanSquaredError:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        # E = (1 / n) * Σ(Y_pred[i] - Y_true[i])^2
        E = np.mean(np.power(y_true - y_pred, 2))
        return E

    @staticmethod
    def mean_squared_error_prime(y_true, y_pred):
        # dE/dY_pred = (2 / n) * (Y_pred - Y_true)
        output_gradient = 2 * (y_pred - y_true) / np.size(y_true)
        return output_gradient


# Binary Cross Entropy loss function and its derivative
class BinaryCrossEntropy:
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        # prevent a possible log(0) and divisions by 0
        epsilon = 10 ** -100
        # E = (-1 / n) * Σ(Y_true[i] * log(Y_pred[i]) + (1 - Y_true[i]) * log(1 - Y_pred[i]))
        E = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return E

    @staticmethod
    def binary_cross_entropy_prime(y_true, y_pred):
        # prevent a possible log(0) and divisions by 0
        epsilon = 10 ** -100
        # dE/dY_pred = (1 / n) * ((1 - Y_true) / (1 - Y_pred) - Y_true / Y_pred)
        output_gradient = ((1 - y_true) / (1 - y_pred + epsilon) - y_true / (y_pred + epsilon)) / np.size(y_true)
        return output_gradient


# Categorical Cross Entropy loss function and its derivative
class CategoricalCrossEntropy:
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        # prevent a possible log(0)
        epsilon = 10 ** -100
        # E = -Σ(Y_true[i] * log(Y_pred[i]))
        E = -np.sum(y_true * np.log(y_pred + epsilon))
        return E

    @staticmethod
    def categorical_cross_entropy_prime(y_true, y_pred):
        # prevent a possible division by 0
        epsilon = 10 ** -100
        # dE/dY_pred = -Y_true / Y_pred
        output_gradient = -y_true / (y_pred + epsilon)
        return output_gradient


# Sparse Categorical Cross Entropy loss function and its derivative
class SparseCategoricalCrossEntropy:
    @staticmethod
    def sparse_categorical_cross_entropy(y_true, y_pred):
        # create a one hot encoded vector from the integer y_true
        one_hot_encoded = np.zeros_like(y_pred)
        one_hot_encoded[y_true] = 1
        # prevent a possible log(0)
        epsilon = 10 ** -100
        # E = -Σ(one_hot_encoded[i] * log(Y_pred[i]))
        E = -np.sum(one_hot_encoded * np.log(y_pred + epsilon))
        # E = -np.log(y_pred[y_true] + epsilon)
        return E

    @staticmethod
    def sparse_categorical_cross_entropy_prime(y_true, y_pred):
        # create a one hot encoded vector from the integer y_true
        one_hot_encoded = np.zeros_like(y_pred)
        one_hot_encoded[y_true] = 1
        # prevent a possible division by 0
        epsilon = 10 ** -100
        # dE/dY_pred = -one_hot_encoded / Y_pred
        output_gradient = -one_hot_encoded / (y_pred + epsilon)
        # output_gradient = -1 / (y_pred[y_true] + epsilon)
        return output_gradient
