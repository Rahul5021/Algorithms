# Logistic Regression

This folder contains a simple implementation of Logistic Regression using gradient descent, modeled through a custom function. This function calculates the optimal weights for a logistic regression model to classify binary outcomes based on one or more independent variables.

## Overview

Logistic Regression is a classification algorithm that models the probability of a binary outcome. It uses the logistic (sigmoid) function to map linear combinations of features into a range of probabilities [0, 1]. Predictions are made by applying a threshold to these probabilities.

### Model Equation

The logistic regression model is represented as:

$$
P(y=1|X) = \frac{1}{1 + e^{-(b_0 + b_1 \cdot x_1 + b_2 \cdot x_2 + \dots + b_n \cdot x_n)}}
$$

where:
- $P(y=1|X)$: Probability of the positive class
- $x_1, x_2, \dots, x_n$: Independent variables (features)
- $b_0$: Intercept term
- $b_1, b_2, \dots, b_n$: Coefficients of the model for each feature

### Gradient Descent

The `gradient` function uses **Gradient Descent** to optimize the weights. The weights are updated iteratively using the following rule:

$$
\beta_j = \beta_j + \alpha \frac{1}{m} \sum_{i=1}^{m} \left(y_i - h_\beta(x_i)\right) x_{ij}
$$

where:
- $\beta_j$: Current weight for feature $j$
- $\alpha$: Learning rate
- $h_\beta(x_i)$: Predicted probability for sample $i$
- $y_i$: Actual label (0 or 1) for sample $i$
- $x_{ij}$: Value of feature $j$ for sample $i$
- $m$: Total number of samples

## Functions

The implementation consists of two functions:

### `gradient(X, y)`
This function performs logistic regression using gradient descent and returns the model weights.

#### Parameters:
- `X`: A 2D NumPy array of input features.
- `y`: A 1D NumPy array of target labels (binary: 0 or 1).

#### Returns:
- `weights[0]`: Intercept term ($b_0$).
- `weights[1:]`: Coefficients ($b_1, b_2, \dots, b_n$) for each feature.

### `sigmoid(z)`
This helper function calculates the sigmoid of the input, mapping it to the range [0, 1].

#### Parameters:
- `z`: A NumPy array or scalar input.

#### Returns:
- Sigmoid transformation of `z`.

## Usage

### Requirements

- Python 3
- NumPy
- Sklearn

To install the required library, run:

```bash
pip install library_name
```
Replace `library_name` with the actual library name.

### Example Code:

Here’s an example of how to use the `gradient` function:

```bash
# Sample training data
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])  # Binary labels

# Train the model
intercept, coefficients = gradient(X_train, y_train)

# Display results
print("Intercept:", intercept)
print("Coefficients:", coefficients)
```

## Applications:

Logistic Regression is widely used for:
- Binary classification tasks such as spam detection, disease prediction, or fraud detection.
- Estimating probabilities for events in various domains.

## License:

This implementation is provided for educational purposes and is licensed under the MIT License.