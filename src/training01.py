import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#import random forest
from sklearn.ensemble import RandomForestRegressor
#import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
#import pickle to save the model
import joblib


def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV/xlsx file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    #check if the file is either csv or xlsx
    if file_path.endswith('.csv'):
        data=pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data=pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Use .csv or .xlsx")
    
    cols=['WH_CSG_PRESS','WH_PRESS','WH_TEMP','GRAVITY','GROSS_FLUID',
          'SURF_CHOKE','SEP_PRESS']
    data=data[cols]
    X=data.drop('GROSS_FLUID',axis=1)
    y=data['GROSS_FLUID']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def data_cleaning(data):
    """
    Clean the data by filtering data based on specific conditions and filling missing values.
    """
    data=data[data['WH_PRESS']<=3500]
    data=data[data['GROSS_FLUID']>=100]
    data=data[data['SURF_CHOKE']<=128]
    #fill missing values with the median of each column
    data = data.fillna(data.median()).reset_index(drop=True)
    return data

def train_model(X_train, y_train,X_test, y_test):
    """
    Train a Random Forest Regressor model and evaluate its performance.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    X_test (pd.DataFrame): Testing features.
    y_test (pd.Series): Testing target variable.

    Returns:
    RandomForestRegressor: The trained Random Forest model.
    """
    # Create a Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    rf_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    # Save the trained model to a file
    # Use joblib to save the model
    joblib.dump(rf_model, '/home/rianr/pypro/myvenv/virtual-well-test/model/rf_model.pkl')
    
    return rf_model

def main():

    file_path = '/home/rianr/pypro/myvenv/virtual-well-test/data/welltest.csv'
    X_train, X_test, y_train, y_test = load_data(file_path)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data = data_cleaning(train_data)
    test_data=test_data.dropna()
    X_train = train_data.drop('GROSS_FLUID', axis=1)
    y_train = train_data['GROSS_FLUID']
    X_test = test_data.drop('GROSS_FLUID', axis=1)
    y_test = test_data['GROSS_FLUID']
    model = train_model(X_train, y_train,X_test, y_test)

if __name__ == "__main__":
    main()


