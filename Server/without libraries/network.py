def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, x_train, y_train, loss, loss_prime, val_split=None, epochs=1000, learning_rate=0.01, verbose=True):
    validating = val_split and 0 < val_split < 1  # determines whether validation is performed
    if validating:
        train_size = int((1 - val_split) * len(x_train))
        x_val = x_train[train_size:]
        y_val = y_train[train_size:]
        x_train = x_train[0: train_size]
        y_train = y_train[0: train_size]

    for epoch in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            y_hat = predict(network, x)

            # error
            error += loss(y, y_hat)

            # backward
            grad = loss_prime(y, y_hat)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        # calculate validation loss for epoch
        if validating:
            val_error = 0
            for x, y in zip(x_val, y_val):
                # forward
                y_hat = predict(network, x)

                # error
                val_error += loss(y, y_hat)

            val_error /= len(x_val)

        error /= len(x_train)
        if verbose:
            if validating:
                print(f'Epoch {epoch + 1}/{epochs}, loss: {error:.4f} - val loss: {val_error:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{epochs}, loss: {error:.4f}')
