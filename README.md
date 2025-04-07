# Diabetes Prediction Model

This repository contains a machine learning project that uses a neural network to predict the likelihood of diabetes based on medical data. The model is built using Python, leveraging libraries such as `pandas`, `scikit-learn`, `tensorflow`, and `keras-tuner` for data processing, model training, and hyperparameter optimization.

## Overview

The project analyzes a dataset (`diabetes.csv`) with 768 records and 9 features, including `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, and the target variable `Outcome` (0 = no diabetes, 1 = diabetes). A neural network is trained to classify individuals as diabetic or non-diabetic, with an initial validation accuracy of approximately 75.97% after 100 epochs.

### Key Features
- Data preprocessing with standardization using `StandardScaler`.
- A neural network with one hidden layer (32 neurons, ReLU activation) and a sigmoid output layer.
- Hyperparameter tuning using `keras-tuner` to optimize the optimizer (e.g., Adam, SGD).
- Correlation analysis to identify the most predictive features (e.g., `Glucose`, `BMI`).

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `keras`
- `keras-tuner`

Install the required packages using pip:

```bash
pip install numpy pandas scikit-learn tensorflow keras keras-tuner