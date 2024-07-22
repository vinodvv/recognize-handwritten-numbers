from functions import (load_data, normalize_data, create_model, train_model,
                       evaluate_model, predict_and_display_images)

# dataset from local file
data_path = 'data/mnist.npz'

# Load data
x_train, y_train, x_test, y_test = load_data(data_path)

# Normalize data
x_train, x_test = normalize_data(x_train, x_test)

# Create model
model = create_model()

# Train model
train_model(model, x_train, y_train, epochs=3)

# Evaluate model
loss, accuracy = evaluate_model(model, x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy {accuracy}")

# Predict and display images
predict_and_display_images(model)
