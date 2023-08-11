import numpy as np

# Generate random input data
input_data = np.random.randn(10, 1)
target_data = 2 * input_data + np.random.rand()
weight = 0.0
bias = 0.0
learning_rate = 0.01
num_of_epochs = 400


# Define the gradient descent function using vectorized operations
def gradient_descent(inputs, targets, weight, bias, learning_rate):
    num_samples = inputs.shape[0]
    gradient_weight = -2 * np.sum(inputs * (targets - (inputs * weight + bias)))
    gradient_bias = -2 * np.sum(targets - (inputs * weight + bias))

    # Update weight and bias using the calculated gradients
    weight -= learning_rate * (1 / num_samples) * gradient_weight
    bias -= learning_rate * (1 / num_samples) * gradient_bias
    return weight, bias


# Function to calculate predictions
def calculate_predictions(inputs, weight, bias):
    return inputs * weight + bias


# Function to calculate mean squared error loss
def calculate_loss(targets, predictions):
    return np.mean((targets - predictions) ** 2)


# Training loop
for epoch in range(num_of_epochs):
    # Perform gradient descent step and update weight and bias
    weight, bias = gradient_descent(
        input_data, target_data, weight, bias, learning_rate
    )

    # Calculate predictions
    predictions = calculate_predictions(input_data, weight, bias)

    # Calculate mean squared error loss
    loss = calculate_loss(target_data, predictions)

    # Print epoch information
    print(f"Epoch {epoch + 1}, loss: {loss}, weight: {weight}, bias: {bias}")
