# Calories-Burned-Prediction

## Overview
This project aims to predict the number of calories burnt using Multiple Linear Regression (MLR) and Bayesian Linear Regression (BLR) models. The dataset contains features related to exercise and health, and the objective is to tune the models to achieve the best predictive performance.

## Data
Contains two `.csv` files, `exercise.csv` and `calories.csv`. More detailed descriptions are given below:
- The `exercise.csv` has 15,000 pieces of data in total (X).
- The `calories.csv` has 15,000 pieces of data in total (Y).
- You have to merge them and split them into 70:10:20 for training, validation, and testing, respectively.

## Features
- Gender
- Age
- Height
- Weight
- Duration
- Heart Rate
- Body Temperature

## Models

### Multiple Linear Regression (MLR)
MLR is trained using maximum likelihood estimation. Polynomial features are used, and the model is optimized with ridge regularization.

### Bayesian Linear Regression (BLR)
BLR incorporates prior knowledge through Bayesian methods. It also uses polynomial features and regularization.

## Result
The best parameters for both MLR and BLR are determined based on validation MSE.

## Packages
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize
from sklearn.model_selection import KFold

```

## Architecture
```
+--------------------+
|      Start         |
+--------------------+
           |
           v
+--------------------+
|  Data Loading      |
+--------------------+
           |
           v
+--------------------+
|  Data Splitting    |
+--------------------+
           |
           v
+--------------------+
| EDA and Feature    | 
| Selection          | 
+--------------------+
           |
           v
+--------------------+
| Feature            |
| Standardization    |
+--------------------+
           |
           v
+--------------------+
| Polynomial         |
| Transformation     |
+--------------------+
           |
           v
+--------------------+
| Model Definition   |
+--------------------+
           |
           v
+--------------------+
| Model Training &   |
| Evaluation         |
+--------------------+
           |
           v
+--------------------+
| Results Visualization|
+--------------------+
           |
           v
+--------------------+
|        End         |
+--------------------+
```
```
