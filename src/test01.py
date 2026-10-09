import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(file_path):
  data=pd.read_csv(file_path)
  cols=['TEST_DATE','DURATION','WH_CSG_PRESS','WH_PRESS',
        'WH_TEMP','OIL_RATE','GRAVITY','WATER_RATE',
        'GROSS_FLUID','SURF_CHOKE','GL_CHOKE','SEP_PRESS']
  data = data[cols]
  # convert TEST_DATE to datetime
  data['TEST_DATE'] = pd.to_datetime(data['TEST_DATE'])
  #save to a new CSV file
  return data

def create_train_test_split(data, column_to_drop=['GROSS_FLUID','OIL_RATE','WATER_RATE'], target_column='GROSS_FLUID', test_size=0.2, random_state=42):
    X = data.drop(columns=column_to_drop)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def estimate_bhp(data, tvd_ft=6000.0, md_ft=None, tubing_id_in=2.992,
                 water_sg=1.05, water_visc_cp=0.6, roughness_in=0.0006):
    """
    Estimate flowing bottom hole pressure (psig) from wellhead data using the
    single-phase liquid pressure-drop method (Economides et al., Petroleum
    Production Systems, 2nd ed., Ch. 7):

        Pwf = Pwh + dP_PE + dP_F
        dP_PE = rho * TVD / 144                         (hydrostatic, psi)
        dP_F  = 2 * f * rho * u^2 * MD / (gc * D) / 144 (Fanning friction, psi)

    - Oil SG from API gravity: SG = 141.5 / (131.5 + API)
    - Oil + water mixed as no-slip liquid, weighted by water cut
    - Dead-oil viscosity from Beggs & Robinson (1975), using WH_TEMP
    - Fanning friction factor from Chen (1979) explicit equation, 16/Re if laminar

    Gas (formation + lift gas) is ignored, so the result is an upper bound for
    gassy / gas-lifted wells. tvd_ft, md_ft and tubing_id_in are well-specific
    and should be replaced with real completion data.
    """
    md_ft = tvd_ft if md_ft is None else md_ft
    gc = 32.174

    # API gravity: drop non-physical values and fill with median
    api = data['GRAVITY'].where(data['GRAVITY'].between(5, 70))
    api = api.fillna(api.median())
    oil_sg = 141.5 / (131.5 + api)

    gross = data['GROSS_FLUID']
    wc = (data['WATER_RATE'] / gross.where(gross > 0)).fillna(0).clip(0, 1)
    mix_sg = oil_sg * (1 - wc) + water_sg * wc
    rho = 62.4 * mix_sg  # lbm/ft3

    # Beggs-Robinson dead-oil viscosity, cp
    temp = data['WH_TEMP'].where(data['WH_TEMP'].between(40, 350))
    temp = temp.fillna(temp.median())
    x = 10 ** (3.0324 - 0.02023 * api) * temp ** -1.163
    oil_visc = 10 ** x - 1
    mu = oil_visc * (1 - wc) + water_visc_cp * wc

    # Velocity in tubing
    d_ft = tubing_id_in / 12
    area = np.pi * d_ft ** 2 / 4
    u = gross.clip(lower=0) * 5.615 / 86400 / area  # ft/s

    # Reynolds number (field units) and Fanning friction factor
    re = 1488 * rho * u * d_ft / mu
    eps = roughness_in / tubing_id_in
    re_safe = re.where(re > 0, np.nan)
    with np.errstate(invalid='ignore'):  # Chen is only used where Re >= 2100
        chen = -4 * np.log10(eps / 3.7065 - 5.0452 / re_safe *
                             np.log10(eps ** 1.1098 / 2.8257 + (7.149 / re_safe) ** 0.8981))
    f = np.where(re < 2100, 16 / re_safe, 1 / chen ** 2)
    f = pd.Series(f, index=data.index).fillna(0)

    dp_pe = rho * tvd_ft / 144
    dp_f = 2 * f * rho * u ** 2 * md_ft / (gc * d_ft) / 144
    return data['WH_PRESS'] + dp_pe + dp_f


def adding_columns(data):
    if 'GROSS_FLUID' in data.columns and 'WATER_RATE' in data.columns:
        data['WATER_CUT']=data['WATER_RATE']/data['GROSS_FLUID']
    if {'WH_PRESS', 'WH_TEMP', 'GRAVITY', 'GROSS_FLUID', 'WATER_RATE'}.issubset(data.columns):
        data['BHP_EST'] = estimate_bhp(data)
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
    data.to_csv("P:/project/pythonpro/myvenv/virtual-well-test/data/processed_data.csv", index=False)
    data = adding_columns(data)
    print(data.head())
    X_train, X_test, y_train, y_test = create_train_test_split(data)
    train_data = pd.concat([X_train, y_train], axis=1)
    analyze_data(train_data)

if __name__ == "__main__":
    main("/home/rianr/pypro/myvenv/virtual-well-test/data/welltest.csv")
