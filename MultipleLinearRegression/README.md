# Multiple Linear Regression

This folder contains a simple implementation of Multiple Linear Regression, a linear approach to modeling the relationship between a dependent variable $y$ and multiple independent variables $X$.

## Overview

Multiple Linear Regression is an extension of Simple Linear Regression that allows us to model a dependent variable based on multiple features. It fits a hyperplane in the feature space that minimizes the sum of squared errors between the predicted and actual values.

### Model Equation

The model for Multiple Linear Regression is represented as:

$$
y = b_0 + b_1 \cdot x_1 + b_2 \cdot x_2 + \dots + b_n \cdot x_n
$$

where:
- $y$: Predicted output (dependent variable)
- $x_1, x_2, \dots, x_n$: Independent variables (features)
- $b_0$: Intercept term
- $b_1, b_2, \dots, b_n$: Coefficients of the model for each feature

### Formula for Coefficients

To find the optimal coefficients $\beta$ (which include $b_0, b_1, \dots, b_n$), the model uses the **Normal Equation**:

$$
\beta = (X^T X)^{-1} X^T y
$$

where:
- $X$: Matrix of input features with an additional column of 1’s (for the intercept)
- $y$: Vector of target values

## Class Contents

The `MultipleLinearRegression` class has two primary methods:
- **fit(X_train, y_train)**: Calculates the coefficients by fitting the model to the training data.
- **predict(X_test)**: Uses the trained model to predict values for the test data.

## Usage

### Requirements

- Python 3
- NumPy

To install the required libraries, run:

```bash
pip install numpy
