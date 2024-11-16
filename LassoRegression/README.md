# Lasso Regression

This folder contains an example of **Lasso Regression** using the `scikit-learn` library. Lasso Regression is a linear regression technique that incorporates an **L1 regularization term** to handle multicollinearity and perform feature selection.

## Overview

Lasso Regression (Least Absolute Shrinkage and Selection Operator) extends Multiple Linear Regression by adding an **L1 regularization term** to the loss function. This regularization penalizes the absolute values of coefficients, shrinking some of them to exactly zero, thereby helping with feature selection.

### Model Equation

The model equation for Lasso Regression is:

$$
y = b_0 + b_1 \cdot x_1 + b_2 \cdot x_2 + \dots + b_n \cdot x_n
$$

where:
- $y$: Predicted output (dependent variable)
- $x_1, x_2, \dots, x_n$: Independent variables (features)
- $b_0$: Intercept term
- $b_1, b_2, \dots, b_n$: Coefficients of the model for each feature

### Loss Function

Lasso Regression minimizes the following loss function:

$$
J(\beta) = \sum_{i=1}^{m} \left( y_i - \hat{y}_i \right)^2 + \lambda \sum_{j=1}^{n} |b_j|
$$

where:
- $\beta$: Vector of model coefficients including $b_0, b_1, \dots, b_n$
- $m$: Number of samples
- $\lambda$: Regularization parameter controlling the strength of the penalty

## Class Used

This example utilizes the `Lasso` class from the `scikit-learn` library. The main features of the `Lasso` class include:
- Automatic feature selection by shrinking some coefficients to zero
- Flexible tuning of regularization using the parameter `alpha`

## Usage

### Requirements

- Python 3
- NumPy
- scikit-learn

To install the required libraries, run:

```bash
pip install numpy scikit-learn
