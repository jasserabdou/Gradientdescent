# Linear Regression with Gradient Descent

This repository contains Python code examples that demonstrate linear regression using gradient descent. Two different implementations are provided, each showcasing a different approach to achieve the same goal.

## Implementation 1: Basic Gradient Descent

### Overview
The first code example (`basic_gradient_descent.py`) demonstrates linear regression using basic gradient descent. It generates random input data, initializes weight and bias, and then performs gradient descent to optimize these parameters.

### Features
- Random input data generation.
- Manual calculation of gradient and parameter updates.
- Training loop for a specified number of epochs.
- Printing of epoch information and parameter values.

### Usage
To run the code:
```
python gradient_descent.py
```

## Implementation 2: Vectorized Gradient Descent

### Overview
The second code example (`vectorized_gradient_descent.py`) showcases linear regression using vectorized gradient descent. It follows the same process as the first example but utilizes NumPy's vectorized operations for improved efficiency.

### Features
- Random input data generation.
- Vectorized gradient computation and parameter updates.
- Training loop for a specified number of epochs.
- Modularized functions for prediction and loss calculation.
- Printing of epoch information and parameter values.

### Usage
To run the code:
```
python vectorized_gradient_descent.py
```

## Conclusion
Both implementations demonstrate the fundamental concept of linear regression using gradient descent. The second implementation leverages NumPy's vectorized operations for improved performance, making it a more efficient and concise solution.

Feel free to explore and compare the two implementations to gain insights into different coding approaches for the same machine learning task.

---
