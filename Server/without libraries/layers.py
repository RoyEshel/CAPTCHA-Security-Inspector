import numpy as np
from scipy import signal


# Base Layer
class Layer:
    def __int__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # Returns output
        pass

    def backward(self, output_gradient, learning_rate):
        # Updates the layer's learnable parameters
        # Returns input gradient
        pass


# Convolutional Layer
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_height, kernel_width, filters_amount):
        input_depth, input_height, input_width = input_shape
        self.filters_amount = filters_amount
        self.kernels_per_filter = input_depth  # the amount of kernels in each filter is equal to channels amount in input image
        self.input_shape = input_shape
        self.output_shape = (filters_amount, input_height - kernel_height + 1, input_width - kernel_width + 1)
        self.filters_shape = (filters_amount, self.kernels_per_filter, kernel_height, kernel_width)
        self.filters = np.random.randn(*self.filters_shape)  # * is used to unpack the tuple
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        for filter_index in range(self.filters_amount):
            for kernel_index in range(self.kernels_per_filter):
                self.output[filter_index] += \
                    signal.correlate2d(self.input[kernel_index], self.filters[filter_index, kernel_index], 'valid')

        return self.output

    def backward(self, output_gradient, learning_rate):
        # Cross-Correlation (★) is sliding a kernel across an image
        # Convolution (∗) is sliding a flipped (180° rotated) kernel across an image
        # i is the index of the filter, j is the index of the kernel within the filter
        filters_gradient = np.zeros(self.filters_shape)  # dE/dF[i][j] = X[j] ★ (dE/dY[i])
        biases_gradient = output_gradient  # dE/dB[i] = dE/dY[i]
        input_gradient = np.zeros(self.input_shape)  # dE/dX[j] = Σ(i=1, i<filters amount)->(dE/dY[i] ∗full F[i][j])

        for filter_index in range(self.filters_amount):
            for kernel_index in range(self.kernels_per_filter):
                filters_gradient[filter_index, kernel_index] = \
                    signal.correlate2d(self.input[kernel_index], output_gradient[filter_index], 'valid')
                input_gradient[kernel_index] += \
                    signal.convolve2d(output_gradient[filter_index], self.filters[filter_index, kernel_index], 'full')

        self.filters -= learning_rate * filters_gradient
        self.biases -= learning_rate * biases_gradient
        return input_gradient


# Dense Layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias  # Y = W · X + B
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)  # dE/dW = dE/dY · transposed(X)
        bias_gradient = output_gradient  # dE/dB = dE/dY
        input_gradient = np.dot(self.weights.T, output_gradient)  # dE/dX = transposed(W) · dE/dY
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient


# Dropout Layer
class Dropout(Layer):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate
        self.mask = None

    def forward(self, input, train=True):
        self.input = input

        if train:
            self.mask = np.random.binomial(1, 1 - self.drop_rate, input.shape)
        else:
            self.mask = np.ones(input.shape)

        self.output = self.input * self.mask
        return self.output

    def backward(self, output_gradient, learning_rate):
        # dY/dX = mask
        input_gradient = output_gradient * self.mask  # dE/dX = dE/dY * dY/dX
        return input_gradient


# Flatten Layer
class Flatten(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (-1, 1)

    def forward(self, input):
        self.input = input
        self.output = np.reshape(self.input, self.output_shape)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.reshape(output_gradient, self.input_shape)
        return input_gradient


# Reshape Layer
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input = input
        self.output = np.reshape(self.input, self.output_shape)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.reshape(output_gradient, self.input_shape)
        return input_gradient


# MaxPooling Layer
class MaxPooling(Layer):
    def __init__(self, pool_size=(2, 2)):
        self.pool_size = pool_size
        self.selections = None  # contains 1 in the location of max value in each patch and 0 in the rest

    def forward(self, input):
        self.input = input
        self.selections = np.zeros_like(self.input)
        patch_height, patch_width = self.pool_size  # the dimensions of the applied patch
        input_depth, input_height, input_width = self.input.shape  # the dimensions of the input
        output_shape = (input_depth, input_height // patch_height, input_width // patch_width)
        self.output = np.empty(output_shape)

        # iterating over every patch in the input in channel-row-col order
        for channel in range(input_depth):
            for i, row in enumerate(range(0, input_height - input_height % patch_height, patch_height)):
                for j, col in enumerate(range(0, input_width - input_width % patch_width, patch_width)):
                    # get the values of the patch
                    patch = self.input[channel, row: row + patch_height, col: col + patch_width]
                    # find the location of the patch's max value (in relation to the patch)
                    max_row, max_col = np.argmax(patch) // patch_width, np.argmax(patch) % patch_width
                    # find the location of the patch's max value (in relation to the input matrix)
                    max_row, max_col = row + max_row, col + max_col
                    # set the location of the patch's max value to 1 in the selections matrix
                    self.selections[channel, max_row, max_col] = 1
                    # add the patch's max value to the output matrix
                    self.output[channel, i, j] = np.max(patch)

        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros_like(self.input)
        input_gradient_depth, input_gradient_height, input_gradient_width = input_gradient.shape
        patch_height, patch_width = self.pool_size  # the dimensions of the patch

        # iterating over every patch in the input gradient in channel-row-col order
        for channel in range(input_gradient_depth):
            for i, row in enumerate(range(0, input_gradient_height - input_gradient_height % patch_height, patch_height)):
                for j, col in enumerate(range(0, input_gradient_width - input_gradient_width % patch_width, patch_width)):
                    # each patch in the input gradient is set to be the multiplication between the matching patch of the selections matrix and output gradient's value for that patch
                    input_gradient[channel, row: row + patch_height, col: col + patch_width] = \
                        self.selections[channel, row: row + patch_height, col: col + patch_width] * \
                        output_gradient[channel, i, j]

        return input_gradient
