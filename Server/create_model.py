import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from matplotlib import pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os


DATASET_PATH = 'Dataset'
IMAGE_SIZE = 100
CHANNELS = 3  # for colored images


def pickle_the_dataset(dataset_path, pickle_path):
    data = []
    classes = os.listdir(dataset_path)

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)  # path to the current class' directory
        for img_name in os.listdir(class_path):
            try:
                # path to current image
                img_path = os.path.join(class_path, img_name)
                # the np array representation of the image
                img_content = cv2.imread(img_path)
                # Reshape to 100x100
                img_content = cv2.resize(img_content, (IMAGE_SIZE, IMAGE_SIZE))
                data.append((img_content, class_index))
            except Exception as e:
                print(e)

    # save the data to a binary file
    pickle_out = open(pickle_path, 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()


def unpickle_data(pickle_path):
    # load the data from the binary file
    pickle_in = open(pickle_path, 'rb')
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data


def shuffle_data(data, times=5):
    for i in range(times):
        random.shuffle(data)
    return data


def split_data(data):
    train_size = int(len(data) * 0.6)
    val_size = int(len(data) * 0.2)

    train_data = data[: train_size]
    val_data = data[train_size: train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


def create_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=64, kernel_size=(7, 7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, train_data, val_data, epochs=20, batch_size=32):
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for features, label in train_data:
        X_train.append(features)
        y_train.append(label)

    for features, label in val_data:
        X_val.append(features)
        y_val.append(label)

    X_train = np.array(X_train) / 255  # convert to np array and normalizes
    y_train = np.array(y_train)  # convert to np array

    X_val = np.array(X_val) / 255  # convert to np array and normalizes
    y_val = np.array(y_val)  # convert to np array

    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return model, hist


def save_model(model, name, path):
    os.makedirs(path, exist_ok=True)
    model.save(os.path.join(path, f'{name}.h5'))


def display_train_progression(hist):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    epochs = [x for x in range(1, len(train_loss) + 1)]

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(len(epochs), 4))

    # Plot the losses on the first subplot
    axs[0].set_title('Loss', fontsize=20)
    axs[0].plot(epochs, train_loss, label='Train loss')
    axs[0].plot(epochs, val_loss, label='Val loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot the accuracies on the second subplot
    axs[1].set_title('Accuracy', fontsize=20)
    axs[1].plot(epochs, train_acc, label='Train acc')
    axs[1].plot(epochs, val_acc, label='Val acc')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Show the figure
    plt.show()


def test_model(model, test_data):
    success = 0

    X_test = []
    y_test = []

    for features, label in test_data:
        X_test.append(features)
        y_test.append(label)

    X_test = np.array(X_test) / 255  # converts to np array and normalizes

    yhat = model.predict(X_test)
    yhat = np.argmax(yhat, axis=1)

    for i in range(len(X_test)):
        if y_test[i] == yhat[i]:
            success += 1

    return success / len(X_test)


def main():
    # pickle_the_dataset(DATASET_PATH, 'Pickle_Jar/data.pickle')
    data = unpickle_data('Pickle_Jar/data.pickle')
    shuffled_data = shuffle_data(data)
    train_data, val_data, test_data = split_data(shuffled_data)
    model = create_model()
    model, hist = train_model(model, train_data, val_data)
    display_train_progression(hist)
    success_rate = round(test_model(model, test_data), 4)
    print(f'Success percentage in testing: {success_rate * 100}%')
    save_model(model, 'top_model', 'models')
    # print(f'Model Architecture:')
    # model.summary()


if __name__ == '__main__':
    main()
