## MNIST Handwritten Digit Classifier with TensorFlow

This project trains a simple convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset.

### Dependencies:

* TensorFlow (https://www.tensorflow.org/)

* NumPy (https://numpy.org/)

* Matplotlib (https://matplotlib.org/)

### Data:

The project expects the MNIST dataset to be present in a file named data/mnist.npz. You can download the MNIST dataset from various sources online.

### Running the Script:

* Make sure you have the required dependencies installed (tensorflow, numpy, matplotlib).
* Place the mnist.npz file in the data directory within your project.
* Run the script using Python (e.g., python main.py).

### Explanation:

The script performs the following steps:

1. Loads the MNIST dataset using numpy.load.
2. Normalizes the pixel values of the images.
3. Defines a sequential model with:

  * A _Flatten_ layer to convert the 2D images into a 1D vector.
  * Three _Dense_ layers with 128 units each and ReLU activation.
  * A final _Dense_ layer with 10 units and softmax activation for digit classification (0-9).

4. Compiles the model using the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.

5. Trains the model on the training data for 3 epochs.

6. Evaluates the model on the test data and prints the loss and accuracy.

7. Makes predictions on the test data.

8. Visualizes 10 random predictions from the test data, highlighting the predicted and true labels for each image.

### Further Exploration:

* Try increasing the number of epochs to improve accuracy.
* Experiment with different hyperparameters (e.g., number of layers, units per layer).
* Try adding convolutional layers to improve performance.

### Note:

This is a basic example to demonstrate MNIST digit classification with TensorFlow. You can extend this code to build more complex models for various image recognition tasks.