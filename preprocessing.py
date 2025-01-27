import numpy as np
import os
import pandas as pd
import warnings

# make those np.log() warnings fuck off
warnings.simplefilter(action='ignore', category=FutureWarning)

# change this to match your directory structure
# os.chdir('Deep Learning')

def transform(monthly_data):

    # apply suggested transformations
    for column in monthly_data:

        # if column == "sasdate":
        #     continue

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

    # drop transformation row as well as first two days since some are not transformed
    monthly_data = monthly_data.drop([0, 1, 2], axis=0)

    return monthly_data

if __name__ == "__main__":

    # load data
    data = pd.read_csv('data/monthly.csv')

    # drop the first column which pertains to dates
    data = data.drop(data.columns[0], axis=1)

    for column in data:
        if data[column].isnull().sum() >= 40:
            print(column, data[column].isnull().sum())

    # deal with missing vals
    for column in data:
        
        # drop columns with at least 50 missing values
        if data[column].isnull().sum() >= 50:
            data = data.drop(column, axis=1)
        
        # fill in missing data with mean (subject to change)
        elif data[column].isnull().sum() > 0:
            column_mean = data[column].mean()
            data[column] = data[column].fillna(column_mean)

    # save transformed data to a new csv file
    transform(data).to_csv('data/monthly_filled.csv', index=False)
