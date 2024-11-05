# Simple Linear Regression

This folder contains an implementation of Simple Linear Regression, a basic machine learning algorithm for modeling the relationship between two continuous variables. 

## Overview

**Simple Linear Regression** is used to predict the value of a dependent variable $y$ based on the value of an independent variable $x$. The model fits a straight line through data points using the **Least Squares Method**, which minimizes the sum of squared differences between observed and predicted values.

### Formula

The equation of the line is: 

$$
y = mx + c
$$

where:
- $m$: Slope of the line
- $c$: y-intercept
- $x$: Input feature (independent variable)
- $y$: Predicted output (dependent variable)

#### Slope Calculation ($m$)

The slope $m$ is calculated as:

$$
m = \frac{\sum (X_i - \bar{X})(y_i - \bar{y})}{\sum (X_i - \bar{X})^2}
$$

where $X_i$ and $y_i$ are individual values from the independent and dependent variable data, and $\bar{X}$ and $\bar{y}$ are the means of $X$ and $y$.



#### Intercept Calculation ($c$)

The intercept $c$ is calculated as:

$$
c = \bar{y} - m \cdot \bar{X}
$$


### Contents

- **SimpleLinearRegression.ipynb**: Jupyter Notebook with the implementation and example usage of Simple Linear Regression.
- **sample_data.csv**: Example dataset to demonstrate the algorithm.
- **README.md**: Explanation of the algorithm, formulas, and instructions for usage.

### Usage

1. **Open the Notebook**: Launch the `SimpleLinearRegression.ipynb` file in Jupyter Notebook or JupyterLab.
2. **Run the Code**: Follow the code cells to fit the model on a sample dataset and make predictions.
3. **Customize with Data**: Replace `sample_data.csv` with your own dataset if desired.

### Requirements

- Python 3
- NumPy
- Pandas
- sklearn

To install the required libraries, run:
```bash
pip install library_name
```

### Example Code Snippet

```bash
# Import and initialize the model
model = SimpleLinearRegression()

# Fit the model on training data
model.fit(X_train, y_train)

# Predict values on test data
y_pred = model.predict(X_test)
```

### License
This implementation is provided for educational purposes and is licensed under the MIT License.

