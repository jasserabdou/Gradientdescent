import numpy as np

# Generate random input data
input_data = np.random.randn(10, 1)
target_data = 2 * input_data + np.random.rand()
weight = 0.0
bias = 0.0
learning_rate = 0.01


# Define the gradient descent function
def gradient_descent(inputs, targets, weight, bias, learning_rate):
    gradient_weight = 0.0
    gradient_bias = 0.0
    num_samples = inputs.shape[0]

    # Compute gradients for weight and bias using each data sample
    for xi, yi in zip(inputs, targets):
        gradient_weight += -2 * xi * (yi - (weight * xi + bias))
        gradient_bias += -2 * (yi - (weight * xi + bias))

    # Update weight and bias using the calculated gradients
    weight -= learning_rate * (1 / num_samples) * gradient_weight
    bias -= learning_rate * (1 / num_samples) * gradient_bias
    return weight, bias


# Training loop
for epoch in range(400):
    # Perform gradient descent step and update weight and bias
    weight, bias = gradient_descent(
        input_data, target_data, weight, bias, learning_rate
    )

    # Calculate predictions based on updated weight and bias
    predictions = weight * input_data + bias

    # Calculate mean squared error loss
    loss = np.divide(
        np.sum((target_data - predictions) ** 2, axis=0), input_data.shape[0]
    )

    # Print epoch information
    print(f"Epoch {epoch + 1}, loss: {loss}, weight: {weight}, bias: {bias}")
