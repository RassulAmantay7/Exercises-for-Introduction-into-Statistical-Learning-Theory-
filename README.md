# Comparative Analysis of Regression Models & Theoretical Portfolio

## Overview

This project provides a hands-on exploration of fundamental concepts in Statistical Learning Theory. It consists of two main parts:

1.  **A Jupyter Notebook** that systematically compares the performance of six different regression models on a challenging synthetic dataset.
2.  **A comprehensive Portfolio (PDF)**, written in LaTeX, that provides the deep theoretical background for the experiments conducted in the notebook.

## Project Components

### 1. Jupyter Notebook (`2_final_code_jupyter.ipynb`)

This notebook contains the complete Python code for the project. It walks through:
- **Synthetic Data Generation:** Creating a dataset with known properties like non-linearity, multicollinearity, and irrelevant features.
- **Model Training & Evaluation:** Implementing, training, and evaluating each of the six regression models.
- **Visualization:** Generating plots for each model to visually interpret its performance, such as coefficient paths for Lasso or learning curves for Gradient Descent.

### 2. Theoretical Portfolio (`portfolio_final_with_theory.pdf`)

This document provides the academic context for the practical work in the notebook. It connects the experimental results directly to the concepts from the Statistical Learning Theory course, and includes:

- **In-depth Theoretical Foundations:** Detailed explanations for each model.
- **Mathematical Formulas:** Key equations such as the Bias-Variance decomposition for OLS, Ridge, and Gradient Descent.
- **Formal Analysis:** A rigorous discussion of the results, explaining *why* certain models performed better than others by referencing concepts like the Curse of Dimensionality and the principles of regularization.

## Key Concepts Demonstrated

This analysis serves as a practical demonstration of several key theoretical concepts:

- **Empirical Risk Minimization (ERM)**
- **The Bias-Variance Trade-off**
- **Explicit Regularization (L1 & L2)**
- **Implicit Regularization (Early Stopping)**
- **The Curse of Dimensionality**

## Models Implemented

1.  Ordinary Least Squares (OLS)
2.  Ridge Regression
3.  Lasso Regression
4.  Gradient Descent with Early Stopping
5.  k-Nearest Neighbors (k-NN)
6.  Multi-Layer Perceptron (MLP)

## Requirements

The Python code in the Jupyter Notebook requires the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
