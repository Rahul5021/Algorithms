# Ridge Regression

This folder contains a simple implementation of Ridge Regression, a linear regression technique that introduces a regularization term to handle multicollinearity and overfitting.

## Overview

Ridge Regression is an extension of Multiple Linear Regression that adds an **L2 regularization term** to the loss function. This regularization penalizes large coefficients, preventing overfitting and improving model generalization.

### Model Equation

The model for Ridge Regression is represented as:

$$
y = b_0 + b_1 \cdot x_1 + b_2 \cdot x_2 + \dots + b_n \cdot x_n
$$

where:
- $y$: Predicted output (dependent variable)
- $x_1, x_2, \dots, x_n$: Independent variables (features)
- $b_0$: Intercept term
- $b_1, b_2, \dots, b_n$: Coefficients of the model for each feature

### Loss Function

Ridge Regression minimizes the following loss function:

\[
J(\beta) = \sum_{i=1}^{m} \left( y_i - \hat{y}_i \right)^2 + \lambda \sum_{j=1}^{n} \beta_j^2
\]

where:
- $\beta$: Vector of model coefficients including $b_0, b_1, \dots, b_n$
- $m$: Number of samples
- $\lambda$: Regularization parameter (controls the strength of the penalty)

### Formula for Coefficients

The optimal coefficients $\beta$ are computed using the **Ridge Regression Normal Equation**:

$$
\beta = (X^T X + \lambda I)^{-1} X^T y
$$

where:
- $X$: Matrix of input features with an additional column of 1’s (for the intercept)
- $y$: Vector of target values
- $\lambda I$: Regularization term (diagonal matrix with $\lambda$ along the diagonal)

## Class Contents

The `RidgeRegression` class has two primary methods:
- **fit(X_train, y_train, alpha)**: Fits the Ridge Regression model to the training data using a specified regularization parameter $\lambda$ (called `alpha` in the code).
- **predict(X_test)**: Predicts target values for the test data using the trained model.

## Usage

### Requirements

- Python 3
- NumPy
- Sklearn

To install the required libraries, run:

```bash
pip install numpy scikit-learn
```

### Example Code
Below is an example demonstrating how to use the RidgeRegression class in Python.
```bash
import numpy as np
from RidgeRegression import RidgeRegression

# Sample training data
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([5, 7, 9, 11])

# Initialize the Ridge Regression model
model = RidgeRegression()

# Fit the model on the training data with regularization parameter alpha=0.1
model.fit(X_train, y_train, alpha=0.1)

# Sample test data
X_test = np.array([[5, 6], [6, 7]])
y_pred = model.predict(X_test)

# Output predictions
print("Predictions:", y_pred)
```

### Explanation of Attributes

- **intercept_**: The intercept $b_0$ in the regression equation.  
- **coef_**: The coefficients $b_1, b_2, \dots, b_n$ for each feature in the model.

### Tuning the Regularization Parameter

The strength of regularization is controlled by the parameter $\lambda$ (referred to as `alpha` in the code). A higher value of `alpha` increases regularization, shrinking the coefficients closer to zero. Tuning `alpha` is essential to balance bias and variance for optimal performance.

## License

This implementation is provided for educational purposes and is licensed under the MIT License.
