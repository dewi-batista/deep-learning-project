import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Function to create time series windows
def create_time_series_windows(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i + time_steps, :-1].values.flatten())
        y.append(data.iloc[i + time_steps, -1])
    return np.array(X), np.array(y)

# Hyperparameter search function
def hyperparam_search(param_grid, X_train, y_train, n_iter=10):
    model = RandomForestRegressor()
    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_params_

# Train model function
def train_model(X_train, y_train, best_params):
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)
    return model

# Test model function
def test_model(model, X_test, y_test, target_covariate):
    y_pred = model.predict(X_test)
    test_loss = mean_squared_error(y_test, y_pred)
    
    plt.figure()
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title(f'Predictions vs Actual Values, test loss: {np.round(test_loss, 8)}')
    plt.xlabel('Window Index')
    plt.ylabel('Value')
    plt.savefig(f'figures/test_RF_{target_covariate}.pdf')
    print('Test loss:', np.round(test_loss, 8))

if __name__ == "__main__":
    for target_covariate in ['UNRATE', 'CPIAUCSL']:
        data = pd.read_csv(f'data/monthly_filled_{target_covariate}.csv')
        time_steps = 12  # TODO
        X, y = create_time_series_windows(data, time_steps)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
        
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        
        best_params = hyperparam_search(param_grid, X_train, y_train)
        print("Best hyperparameters:", best_params)
        
        model = train_model(X_train, y_train, best_params)
        test_model(model, X_test, y_test, target_covariate)
