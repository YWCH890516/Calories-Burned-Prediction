import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize
from sklearn.model_selection import KFold

# preprocess data----------------------------------------------------------------------------------------------------
# read csv file to this code 
df_ex = pd.read_csv('exercise.csv')
df_cal= pd.read_csv('calories.csv')

# check the data which is already read in this code 
print(df_ex.head())
print('-'*500)
print(df_cal.head())

# merge two file 
df_merged = pd.merge(df_ex, df_cal, on='User_ID')
df_merged.replace({'male': 0, 'female': 1}, inplace=True)
df_merged = df_merged.fillna(df_merged.mean())
print(df_merged.head()) 

# 80% train+val 20%test
train_val_set, test_set = train_test_split(df_merged, test_size=0.2, random_state=42)
# 70% train 10%val
train_set, val_set = train_test_split(train_val_set, test_size=0.125, random_state=42)

print(f"training set size : {len(train_set)}")
print(f"val set size : {len(val_set)}")
print(f"testing set size : {len(test_set)}")
"""
# see the data disstribution ------ refer to: https://blog.csdn.net/qq_42034590/article/details/131461436
sb.scatterplot(x=df_merged['Height'],y=df_merged['Weight'])
plt.show()
###### weight and height are linear relationship
features = ['Age','Height','Weight','Duration']
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    X = df_merged.sample(1000)
    sb.scatterplot(x=X[col], y=X['Calories'])
plt.tight_layout()
plt.show()
features = df_merged.select_dtypes(include='float').columns
  
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(df_merged[col], kde=True)
plt.tight_layout()
plt.show()

# heat map 
plt.figure(figsize = (6,6))
sb.heatmap(df_merged.corr()>0.9 , annot =True, cbar=False)
plt.show()  # -----------> 

remove_item = ['Weight','Duration']
df_merged.drop(remove_item,axis=1,inplace=True)
"""
#----------------------------Define Function----------------------------------------------------------------------------

# Define the negative log-likelihood function for regression 
# reference: https://learningdaily.dev/understanding-maximum-likelihood-estimation-in-machine-learning-22b915c3e05a
# Function to standardize features

def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - mean) / std
    return X_std, mean, std

def neg_log_likelihood(theta, X, y, alpha=1.0):
    z = np.dot(X, theta)
    res = y - z 
    sigma = np.std(res)
    n_log_likelihood = 0.5 * len(y) * np.log(2 * np.pi * sigma**2) + np.sum(res**2) / (2 * sigma**2)
    ridge_penalty = alpha * np.sum(theta**2)
    return n_log_likelihood + ridge_penalty

# Multiple Linear Regression using Maximum Likelihood Estimation
def MLR(X_train, y_train, X_test, y_test, degree=1, alpha=1.0):
    X_train_std, mean, std = standardize_features(X_train)
    X_test_std = (X_test - mean) / std
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train_std)
    X_test_poly = poly.transform(X_test_std)
    w = np.linalg.inv(X_train_poly.T.dot(X_train_poly) + alpha * np.eye(X_train_poly.shape[1])).dot(X_train_poly.T).dot(y_train)
    y_train_pred = X_train_poly.dot(w)
    y_test_pred = X_test_poly.dot(w)
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    return w, train_mse, test_mse, y_train_pred, y_test_pred

def BLR(X_train, y_train, X_test, y_test, degree=1, alpha=1.0, beta=1.0):
    # Standardize features
    X_train_std, mean, std = standardize_features(X_train)
    X_test_std = (X_test - mean) / std
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train_std)
    X_test_poly = poly.transform(X_test_std)
    # Compute prior and posterior distribution
    X_train_poly_T_X_train_poly = X_train_poly.T.dot(X_train_poly)
    S_N_inv = alpha * np.eye(X_train_poly.shape[1]) + beta * X_train_poly_T_X_train_poly
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(X_train_poly.T).dot(y_train)  
    # Predict
    y_train_pred = X_train_poly.dot(m_N)
    y_test_pred = X_test_poly.dot(m_N)
    
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    
    return m_N, S_N, train_mse, test_mse, y_train_pred, y_test_pred
def train_and_evaluate(X_train, y_train, X_val, y_val, degree_range, alpha_range):
    best_mse = float('inf')
    best_params = None
    
    
    kf = KFold(n_splits=5)
    
    for degree in degree_range:
        for alpha in alpha_range:
            val_mses = []
            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                _, _, val_mse, _, _ = MLR(X_train_fold, y_train_fold, X_val_fold, y_val_fold, degree, alpha)
                val_mses.append(val_mse)
                
            avg_val_mse = np.mean(val_mses)
            if avg_val_mse < best_mse:
                best_mse = avg_val_mse
                best_params = (degree, alpha)
    
    return best_params, best_mse

def train_and_evaluate_blr(X_train, y_train, X_val, y_val, degree_range, alpha_range, beta_range):
    best_mse = float('inf')
    best_params = None
    
    
    kf = KFold(n_splits=5)
    
    for degree in degree_range:
        for alpha in alpha_range:
            for beta in beta_range:
                val_mses = []
                for train_index, val_index in kf.split(X_train):
                    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                    _, _, _, val_mse, _, _ = BLR(X_train_fold, y_train_fold, X_val_fold, y_val_fold, degree, alpha, beta)
                    val_mses.append(val_mse)
                
                avg_val_mse = np.mean(val_mses)
                if avg_val_mse < best_mse:
                    best_mse = avg_val_mse
                    best_params = (degree, alpha, beta)
    
    return best_params, best_mse

def main():
    features = ['Age', 'Height', 'Gender','Heart_Rate','Body_Temp','Duration']
    X_train = train_set[features].values
    y_train = train_set['Calories'].values
    X_val = val_set[features].values
    y_val = val_set['Calories'].values
    X_test = test_set[features].values
    y_test = test_set['Calories'].values
    
    # MLR hyperparameter tuning
    degree_range = range(1, 4)
    alpha_range = [0.001, 0.01, 0.1, 1.0]
    best_mlr_params, best_mlr_mse = train_and_evaluate(X_train, y_train, X_val, y_val, degree_range, alpha_range)
    print(f"Best MLR Parameters: {best_mlr_params}, Best MSE: {best_mlr_mse}")

    degree, alpha = best_mlr_params
    _, train_mse, val_mse, _, y_val_pred = MLR(X_train, y_train, X_val, y_val, degree, alpha)
    print(f"Training MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")

    _, test_mse, _, _, y_test_pred = MLR(X_train, y_train, X_test, y_test, degree, alpha)
    print(f"Testing MSE: {test_mse}")

    plt.figure(figsize=(12, 6))
    plt.scatter(y_val, y_val_pred, color='blue', label='Validation Set')
    plt.scatter(y_test, y_test_pred, color='red', label='Test Set')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('Actual Calories Burnt')
    plt.ylabel('Predicted Calories Burnt')
    plt.title('Actual vs Predicted Calories Burnt')
    plt.legend()
    plt.show()

    # BLR hyperparameter tuning
    beta_range = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_blr_params, best_blr_mse = train_and_evaluate_blr(X_train, y_train, X_val, y_val, degree_range, alpha_range, beta_range)
    print(f"Best BLR Parameters: {best_blr_params}, Best MSE: {best_blr_mse}")

    degree, alpha, beta = best_blr_params
    _, _, train_mse2, val_mse2, _, y_val_pred2 = BLR(X_train, y_train, X_val, y_val, degree, alpha, beta)
    print(f"Training MSE: {train_mse2}")
    print(f"Validation MSE: {val_mse2}")

    _, _, test_mse2, _, _, y_test_pred2 = BLR(X_train, y_train, X_test, y_test, degree, alpha, beta)
    print(f"Testing MSE: {test_mse2}")

    plt.figure(figsize=(12, 6))
    plt.scatter(y_val, y_val_pred2, color='blue', label='Validation Set')
    plt.scatter(y_test, y_test_pred2, color='red', label='Test Set')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('Actual Calories Burnt')
    plt.ylabel('Predicted Calories Burnt')
    plt.title('Actual vs Predicted Calories Burnt')
    plt.legend()
    plt.show()

# Execute the main function
if __name__ == "__main__":
    main()
