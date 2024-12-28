# Gradient Boosting Regressor from Scratch

This project demonstrates the implementation of a Gradient Boosting Regressor from scratch using Python. The notebook contains:

- A custom implementation of Gradient Boosting.
- Use of decision trees as base learners.
- Training and evaluation on regression datasets.

## Folder Structure

- `GradientBoosting.ipynb`: The main Jupyter Notebook where the Gradient Boosting Regressor is implemented and tested.

## Getting Started

To run the notebook, follow these steps:

### Prerequisites

Ensure you have Python installed along with the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`

Install them using pip if not already installed:
```bash
pip install numpy pandas scikit-learn
```

### Running the Notebook

1. Open the Jupyter Notebook:
```bash
jupyter notebook GradientBoosting.ipynb
```

2. Execute the cells step-by-step to understand and observe the custom implementation of the Gradient Boosting Regressor.

## Key Features

- **Custom Gradient Boosting Implementation**: The algorithm is built from scratch using Python without relying on pre-built Gradient Boosting libraries.
- **Residual Calculation**: Residuals are calculated at each step to improve the model's predictions iteratively.
- **Decision Trees as Base Learners**: Decision trees are utilized as weak learners to build the ensemble.


## Results

The implementation is tested on regression datasets, and residual analysis is performed to evaluate the model's performance. For more details, explore the notebook.

## License

This project is licensed under the MIT License - see the LICENSE file for details