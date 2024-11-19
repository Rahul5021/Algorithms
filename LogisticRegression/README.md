# Logistic Regression

This folder contains a simple implementation of Logistic Regression, a classification algorithm used to model the probability of a binary outcome based on one or more independent variables.

## Overview

Logistic Regression is a statistical method for binary classification. Instead of predicting a continuous outcome like linear regression, logistic regression predicts the probability of the target belonging to a particular class. It is based on the logistic (sigmoid) function, which maps any real-valued number into the range [0, 1].

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

The predicted class is determined by applying a threshold (e.g., 0.5) to the predicted probability.

### Cost Function

Logistic Regression minimizes the **Log Loss**:

$$
J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_\beta(x_i)) + (1 - y_i) \log(1 - h_\beta(x_i)) \right]
$$

where:
- $m$: Number of samples
- $h_\beta(x_i)$: Predicted probability for sample $i$
- $y_i$: Actual class label (0 or 1) for sample $i$

### Optimization

The coefficients $\beta$ are optimized using **Gradient Descent**:

$$
\beta_j = \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}
$$

where:
- $\alpha$: Learning rate
- $\frac{\partial J(\beta)}{\partial \beta_j}$: Partial derivative of the cost function with respect to $\beta_j$

## Usage

### Requirements

- Python 3
- NumPy
- Sklearn

To install the required libraries, run:

```bash
pip install library_name
```
Replace `library_name` with the actual library name.

## Applications:

Logistic Regression is widely used for:
- Binary classification tasks such as spam detection, disease prediction, or fraud detection.
- Estimating probabilities for events in various domains.

## License:

This implementation is provided for educational purposes and is licensed under the MIT License.