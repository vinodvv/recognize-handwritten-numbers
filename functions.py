import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data(data_path):
    """
    Load the MNIST data set from the given path.

    Args:
        data_path (str): Path to the mnist.npz file.

    :return:
        tuple: Tuple containing training and test data.
    """

    with np.load(data_path, allow_pickle=True) as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
    return x_train, y_train, x_test, y_test


def normalize_data(x_train, x_test):
    """
    Normalize the training and test data.

    :param x_train: x_train (numpy.ndarray): Training data.
    :param x_test:  y_train (numpy.ndarray): Test data.
    :return:
        tuple: Tuple containing normalized training and test data.
    """

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    return x_train, x_test


def create_model():
    """
    Create and compile a neural network model.

    :return:
        tensorflow.keras.Model: Compiled neural network model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, epochs=3):
    """
    Train the neural network model.

    :param model: model (tensorflow.keras.Model): Neural network model.
    :param x_train: x_train (numpy.ndarray): Training data.
    :param y_train: y_train (numpy.ndarray): Training labels.
    :param epochs: epochs (int): Number of epochs to train the model
    """
    model.fit(x_train, y_train, epochs=epochs)


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the neural network model.

    :param model: (tensorflow.keras.Model): Neural network model.
    :param x_test: (numpy.ndarray): Test data.
    :param y_test: (numpy.ndarray): Test labels.
    :return:
        tuple: Tuple containing loss and accuracy.
    """
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy


def make_predictions(model, x_test):
    """
    Make predictions using the trained model.

    :param model: (tensorflow.keras.Model): Trained neural network model.
    :param x_test: (numpy.ndarray): Test data.
    :return:
        numpy.ndarray: Predications.
    """
    return model.predict(x_test)


def visualize_predictions(x_test, y_test, predictions, num_images=10):
    """
    Visualize the predictions made by the model.

    :param x_test: (numpy.ndarray): Test data.
    :param y_test: (numpy.ndarray): Test labels.
    :param predictions: (numpy.ndarray): Predictions made by the model.
    :param num_images: (int): Number of images t visualize. Default is 10. Maximum is 25.
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i], cmap="gray")
        predicted_label = np.argmax(predictions[i])
        true_label = y_test[i]
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel(f"{predicted_label} ({true_label})", color=color)
    plt.show()


def predict_and_display_images(model, image_folder='digits'):
    """
    Predict and display images from the given folder using the trained model.

    :param model: (tensorflow.keras.Model): Trained neural network model.
    :param image_folder: (str): Folder containing images to predict.
    """
    image_number = 1
    while os.path.isfile('digits/{}.png'.format(image_number)):
        try:
            img = cv2.imread('digits/{}.png'.format(image_number))[:, :, 0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print("The number is probably a {}".format(np.argmax(prediction)))
            plt.imshow(img[0], cmap="gray")
            plt.show()
        except:
            print("Error reading image! Proceeding to the next one...")
        finally:
            image_number += 1
