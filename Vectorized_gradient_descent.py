import numpy as np


def generate_data(num_samples: int, noise: float = 0.1) -> tuple:
    """
    Generate random input and target data for linear regression.

    :param num_samples: Number of samples to generate.
    :param noise: Amount of noise to add to the target data.
    :return: Tuple containing the input data and target data.
    """
    input_data = np.random.randn(num_samples, 1)
    target_data = 2 * input_data + noise * np.random.randn(num_samples, 1)
    return input_data, target_data


def gradient_descent(
    inputs: np.ndarray,
    targets: np.ndarray,
    weight: float,
    bias: float,
    learning_rate: float,
) -> tuple:
    """
    Perform a single step of gradient descent on a linear regression model.

    :param inputs: Input data.
    :param targets: Target data.
    :param weight: Current weight of the model.
    :param bias: Current bias of the model.
    :param learning_rate: Learning rate to use for gradient descent.
    :return: Tuple containing the updated weight and bias.
    """
    num_samples = inputs.shape[0]
    gradient_weight = (
        -2 * np.sum(inputs * (targets - (inputs * weight + bias))) / num_samples
    )
    gradient_bias = -2 * np.sum(targets - (inputs * weight + bias)) / num_samples
    weight -= learning_rate * gradient_weight
    bias -= learning_rate * gradient_bias
    return weight, bias


def calculate_predictions(inputs: np.ndarray, weight: float, bias: float) -> np.ndarray:
    """
    Calculate predictions for a linear regression model.

    :param inputs: Input data.
    :param weight: Weight of the model.
    :param bias: Bias of the model.
    :return: Predictions for the input data.
    """
    return inputs * weight + bias


def calculate_loss(targets: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate the mean squared error loss for a linear regression model.

    :param targets: Target data.
    :param predictions: Predictions for the input data.
    :return: Mean squared error loss.
    """
    return np.mean((targets - predictions) ** 2)


def train_linear_regression(
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float = 0.01,
    num_epochs: int = 400,
) -> tuple:
    """
    Train a linear regression model using gradient descent.

    :param inputs: Input data.
    :param targets: Target data.
    :param learning_rate: Learning rate to use for gradient descent.
    :param num_epochs: Number of epochs to train for.
    :return: Tuple containing the trained weight and bias of the model.
    """
    weight = 0.0
    bias = 0.0
    for epoch in range(num_epochs):
        weight, bias = gradient_descent(inputs, targets, weight, bias, learning_rate)
        predictions = calculate_predictions(inputs, weight, bias)
        loss = calculate_loss(targets, predictions)
        print(
            f"Epoch {epoch + 1}, loss: {loss:.4f}, weight: {weight:.4f}, bias: {bias:.4f}"
        )
    return weight, bias


# Generate random input and target data
num_samples = 100
input_data, target_data = generate_data(num_samples)

# Train the linear regression model
trained_weight, trained_bias = train_linear_regression(input_data, target_data)
