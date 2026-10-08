import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(file_path):
  data=pd.read_csv(file_path)
  cols=['TEST_DATE','DURATION','WH_CSG_PRESS','WH_PRESS',
        'WH_TEMP','OIL_RATE','GRAVITY','WATER_RATE',
        'GROSS_FLUID','SURF_CHOKE','SEP_PRESS']
  data = data[cols]
  # convert TEST_DATE to datetime
  data['TEST_DATE'] = pd.to_datetime(data['TEST_DATE'])
  #save to a new CSV file
  return data

def create_train_test_split(data, column_to_drop=['GROSS_FLUID','OIL_RATE','WATER_RATE'], target_column='GROSS_FLUID', test_size=0.2, random_state=42):
    X = data.drop(columns=column_to_drop)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def adding_columns(data):
    if 'GROSS_FLUID' in data.columns and 'WATER_RATE' in data.columns:
        data['WATER_CUT']=data['WATER_RATE']/data['GROSS_FLUID']
    return data


def analyze_data(data):
    # checking data types
    print(data.dtypes)
    # checking for missing values
    print(data.isnull().sum())
    #check percentage of missing values
    print((data.isnull().sum() / len(data)) * 100)
    # plot histogram for each numeric column and images saved as png instead of showing
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    print(data[numeric_cols].describe())
    data[numeric_cols].hist(figsize=(15, 10))
    plt.savefig("/home/rianr/pypro/myvenv/virtual-well-test/Images/numeric_histograms.png")
    plt.close()
    #plot correlation matrix and also show the values on the matrix
    corr = data[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    plt.matshow(corr, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')
    plt.savefig("/home/rianr/pypro/myvenv/virtual-well-test/Images/correlation_matrix.png")
    plt.close()


def main(file_path):
    data = load_data(file_path)
    #making all columns when printed
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    data.to_csv("/home/rianr/pypro/myvenv/virtual-well-test/data/processed_data.csv", index=False)
    data = adding_columns(data)
    print(data.head())
    X_train, X_test, y_train, y_test = create_train_test_split(data)
    train_data = pd.concat([X_train, y_train], axis=1)
    analyze_data(train_data)

if __name__ == "__main__":
    main("/home/rianr/pypro/myvenv/virtual-well-test/data/welltest.csv")
