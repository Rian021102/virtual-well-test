import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_data(file_path):
  data=pd.read_csv(file_path)
  cols=['TEST_DATE','DURATION','WH_CSG_PRESS','WH_PRESS',
        'WH_TEMP','OIL_RATE','GRAVITY','WATER_RATE','FM_GAS',
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

def choke_gilbert(data):
    # Placeholder function for choke_gilbert calculation
    return data

def adding_columns(data):
    if 'GROSS_FLUID' in data.columns and 'WATER_RATE' in data.columns:
        # avoid division by zero: rows with GROSS_FLUID == 0 become NaN
        data['WATER_CUT']=data['WATER_RATE']/data['GROSS_FLUID']
        data['GAS_LIQUID_RATIO'] = np.where(data['FM_GAS'] > 0, (data['FM_GAS']*1000) / data['GROSS_FLUID'], 0)
        data[['WATER_CUT', 'GAS_LIQUID_RATIO']] = data[['WATER_CUT', 'GAS_LIQUID_RATIO']].replace([np.inf, -np.inf], np.nan)
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
    print("Number of WH pressure more than 3500 Psi: ", (data['WH_PRESS'] > 3500).sum())
    print("Number of SURF_CHOKE greater than 128: ", (data['SURF_CHOKE'] > 128).sum())
    data[numeric_cols].hist(figsize=(15, 10))
    # check if the histogram image already exists next save would be the _1, _2, etc.
    histogram_path = "/home/rianr/pypro/myvenv/virtual-well-test/Images/numeric_histograms.png"
    if os.path.exists(histogram_path):
        base, ext = os.path.splitext(histogram_path)
        i = 1
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        histogram_path = f"{base}_{i}{ext}"
    plt.savefig(histogram_path)
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
    corr_path = "/home/rianr/pypro/myvenv/virtual-well-test/Images/correlation_matrix.png"
    if os.path.exists(corr_path):
        base, ext = os.path.splitext(corr_path)
        i = 1
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        corr_path = f"{base}_{i}{ext}"
    plt.savefig(corr_path)
    plt.close()

def data_cleaning(data, medians=None):
    #drop WH_PRESS values greater than 3500 Psi
    data = data[data['WH_PRESS'] <= 3500]
    # data=data[data['GAS_LIQUID_RATIO'] <= 1000]
    data=data[data['GROSS_FLUID'] >=100]
    data=data[data['SURF_CHOKE'] <= 128]
    #impute missing values with median (pass the train medians when cleaning test data)
    if medians is None:
        medians = data.median()
    data = data.fillna(medians)
    return data, medians

def train_model(X_train,y_train,X_test,y_test):
    #train random forest regressor and xgboost regressor separately and compare them
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    predictions = {}
    r2_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        #evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - Mean Squared Error: {mse}")
        print(f"{name} - R^2 Score: {r2}")
        predictions[name] = y_pred
        r2_scores[name] = r2
    #plot the r2 score for both random forest and xgboost
    plt.figure(figsize=(8, 6))
    plt.bar(list(r2_scores.keys()), list(r2_scores.values()))
    plt.ylim(0, 1)
    plt.title('R^2 Score')
    plt.savefig("/home/rianr/pypro/myvenv/virtual-well-test/Images/r2_score.png")
    plt.close()
    #plot feature of importance for the random forest regressor and xgboost regressor
    rf_importances = models['Random Forest'].feature_importances_
    xgb_importances = models['XGBoost'].feature_importances_
    feature_names = X_train.columns

    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, rf_importances, alpha=0.6, label='Random Forest')
    plt.bar(feature_names, xgb_importances, alpha=0.6, label='XGBoost')
    plt.xticks(rotation=90)
    plt.title('Feature Importances')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/rianr/pypro/myvenv/virtual-well-test/Images/feature_importances.png")
    plt.close()

    return predictions, models



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
    test_data = pd.concat([X_test, y_test], axis=1)
    analyze_data(train_data)
    #clean the data
    train_data, train_medians = data_cleaning(train_data)
    print('===================================After Cleaning===================================')
    analyze_data(train_data)
    print('Length of train_data after cleaning: ', len(train_data))
    # separate cleaned train data back into features and target
    X_train = train_data.drop(columns=['GROSS_FLUID'])
    y_train = train_data['GROSS_FLUID']
    # cleaned test data
    test_data = test_data.dropna()
    X_test = test_data.drop(columns=['GROSS_FLUID'])
    y_test = test_data['GROSS_FLUID']
    # models can't use datetime columns, drop TEST_DATE from the features
    # WATER_CUT and GAS_LIQUID_RATIO are computed from GROSS_FLUID (the target), drop them to avoid leakage
    leak_cols = ['TEST_DATE', 'WATER_CUT', 'GAS_LIQUID_RATIO']
    X_train = X_train.drop(columns=leak_cols)
    X_test = X_test.drop(columns=leak_cols)


    predictions, models = train_model(X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    main("/home/rianr/pypro/myvenv/virtual-well-test/data/welltest.csv")
