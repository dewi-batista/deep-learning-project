from sklearn.impute import KNNImputer

import numpy as np
import pandas as pd
import warnings

# make those np.log() warnings fuck off
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", category=FutureWarning)

# this could benefit from a switch statement, not sure if that's a thing in python though
def transform(monthly_data):

    # apply the advised de-trending transformations
    for column in monthly_data:

        column_vals = monthly_data[column][1:]

        if monthly_data[column][0] == 1:
            continue

        if monthly_data[column][0] == 2:
            for row_idx in range(column_vals.shape[0], 1, -1):
                monthly_data.loc[row_idx, column] = column_vals[row_idx] - column_vals[row_idx - 1]

        if monthly_data[column][0] == 3:
            for row_idx in range(column_vals.shape[0], 2, -1):
                monthly_data.loc[row_idx, column] = (column_vals[row_idx] - column_vals[row_idx - 1]) - (column_vals[row_idx - 1] - column_vals[row_idx - 2])

        if monthly_data[column][0] == 4:
            for row_idx in range(column_vals.shape[0], 0, -1):
                monthly_data.loc[row_idx, column] = np.log(column_vals[row_idx])

        if monthly_data[column][0] == 5:
            for row_idx in range(column_vals.shape[0], 1, -1):
                monthly_data.loc[row_idx, column] = np.log(column_vals[row_idx]) - np.log(column_vals[row_idx - 1])

        if monthly_data[column][0] == 6:
            for row_idx in range(column_vals.shape[0], 2, -1):
                monthly_data.loc[row_idx, column] = np.log(column_vals[row_idx]) - np.log(column_vals[row_idx - 1])
                monthly_data.loc[row_idx, column] -= np.log(column_vals[row_idx - 1]) - np.log(column_vals[row_idx - 2])

        if monthly_data[column][0] == 7:
            for row_idx in range(column_vals.shape[0], 1, -1):
                monthly_data.loc[row_idx, column] = column_vals[row_idx] / column_vals[row_idx - 1] - 1

    # drop transformation row as well as first two months since some have values that are not transformed
    monthly_data = monthly_data.drop([0, 1, 2], axis=0)

    return monthly_data

if __name__ == "__main__":

    # choose target_covariate (UNRATE or CPIAUCSL)
    target_covariate = 'UNRATE'

    # load data
    data = pd.read_csv('data/monthly.csv')

    # drop the first column (which pertains to dates) and make target_covariate the final column
    data = data.drop(data.columns[0], axis=1)
    data[target_covariate] = data.pop(target_covariate)

    # 1) drop columns with at least 50 missing values and
    #    fill in remaining missing data via kNN with k = 5
    for column in data:
        if data[column].isnull().sum() >= 50:
            data = data.drop(column, axis=1)
    data = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(data), columns=data.columns)

    # 2) apply advised de-trending transformations
    data = transform(data)

    # 3) add lags to data (hopefully this resolves having so little data)
    features = data.iloc[:, :-1]
    lag_num = 3
    for lag in range(1, lag_num + 1):
        for col in features.columns:
            data[f"{col}_L{lag}"] = features[col].shift(lag)
    data = data.iloc[lag_num:] # some missing values after introducing lags

    # 4) remove features that present low correlation with target_covariate
    data[target_covariate] = data.pop(target_covariate)
    correlations = data.corr()[target_covariate].abs()
    high_corr_features = correlations[correlations > 0.2].index
    data = data[high_corr_features]

    # save to a new csv file
    data.to_csv(f'data/monthly_{target_covariate}.csv', index=False)
