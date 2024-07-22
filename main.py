from functions import (load_data, normalize_data, create_model, train_model,
                       evaluate_model, make_predictions, visualize_predictions)

# dataset from the local file
data_path = 'data/mnist.npz'

# Load the dataset
x_train, y_train, x_test, y_test = load_data(data_path)

# Normalize the data
x_train, x_test = normalize_data(x_train, x_test)

# Define the model and compile
model = create_model()

# Train the model
train_model(model, x_train, y_train, epochs=3)

# Evaluate the model
loss, accuracy = evaluate_model(model, x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy {accuracy}")

# Make predictions
predictions = make_predictions(model, x_test)

# Visualize some predictions
visualize_predictions = visualize_predictions(x_test, y_test, predictions, num_images=10)
