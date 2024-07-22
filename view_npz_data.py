"""
To view the NPZ dataset images before starting with the project, you can use the following Python script.
It will display the 34th image of the training data. You can change the "33" index in the code to another number
to see another image.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from the local file
data_path = "data/mnist.npz"

with np.load(data_path, allow_pickle=True) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

# Display the first training image and its label.
plt.imshow(x_train[33], cmap='gray')
plt.title(f"Label: {y_train[33]}")
plt.colorbar()
plt.grid(False)
plt.show()
