"""
Training script: Random Forest liquid-rate model from wellhead data, following IPTC-23662-MS
(Setiadi et al., 2024). Bottom-hole temperature (est. BHT from reservoir depth) is not used
because no depth data is available.

Model selection and analysis (algorithm comparison, correlation, plots, unseen-well check) live
in test01.py, which reuses the data preparation defined here. Random Forest was selected there;
RF_PARAMS holds its tuned hyper-parameters.

Workflow:
  1. Load well tests and derive the paper's variables:
       QG             gas rate = gas lift injection + formation gas (MSCFPD)
       WATER_CUT_FRAC water cut from the test (in deployment: from fluid sampling)
       QLIQ_EQ        mass-equivalent liquid rate = oil*SG_oil + water*1.0 (target)
  2. Clean: only real WHT readings, valid ranges, remove commingled tests.
  3. Split 80:20 (default: random, as in the paper; 'time' holds out each string's latest 20%
     of tests as blind data) and evaluate.
  4. Combined regression: Random Forest sensitivity sweep on WH_TEMP + polynomial fit, so the
     prediction keeps a trend outside the training range instead of going flat.
  5. Refit on all clean data and save the model bundle.

At deployment QG is not measured during a test; use GL_RATE + GOR(last test) * oil estimate,
as the paper does. The paper found the prediction is not sensitive to that assumption.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'welltest.csv'
MODEL_PATH = ROOT / 'model' / 'rf_model.pkl'

RAW_COLS = ['WELL_NAME', 'STRING_CODE', 'TEST_DATE', 'WH_CSG_PRESS', 'WH_PRESS', 'WH_TEMP',
            'GRAVITY', 'OIL_RATE', 'WATER_RATE', 'GROSS_FLUID', 'SURF_CHOKE', 'SEP_PRESS',
            'GL_RATE', 'GL_CHOKE', 'FM_GAS', 'TOTAL_GAS']
# Paper features (WHFP, WHFT, Qg, watercut) plus choke, casing (gas-lift) pressure,
# separator pressure and oil gravity, which are available daily on this field.
FEATURES = ['WH_PRESS', 'WH_TEMP', 'QG', 'WATER_CUT_FRAC', 'SURF_CHOKE', 'WH_CSG_PRESS',
            'SEP_PRESS', 'GRAVITY']
TARGET = 'QLIQ_EQ'
SWEEP_FEATURE = 'WH_TEMP'
WATER_SG = 1.0
POLY_DEGREE = 2

# Tuned in test01.py (grid search with time-block CV on the training set).
RF_PARAMS = {'n_estimators': 300, 'max_features': 0.5, 'min_samples_leaf': 2}

# Strings whose tests flow through a shared line before the test header (paper: commingled
# production). There is no flag for this in the data; fill in from operations knowledge.
COMMINGLED_STRINGS = []


# ---------------------------------------------------------------------------------------------
# Data preparation (shared with test01.py)
# ---------------------------------------------------------------------------------------------
def load_data(file_path):
    """Load raw well tests from .csv or .xlsx and keep the columns the workflow needs."""
    file_path = str(file_path)
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, low_memory=False)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Use .csv or .xlsx")
    missing = [c for c in RAW_COLS if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    data = data[RAW_COLS].copy()
    data['TEST_DATE'] = pd.to_datetime(data['TEST_DATE'], errors='coerce')
    return data


def oil_specific_gravity(api):
    return 141.5 / (131.5 + api)


def equivalent_factor(gravity, water_cut):
    """QLIQ_EQ / GROSS_FLUID: liquid density relative to water for a given API and water cut."""
    return (1 - water_cut) * oil_specific_gravity(gravity) + water_cut * WATER_SG


def add_derived_columns(data):
    """Add QG, WATER_CUT_FRAC and the mass-equivalent target QLIQ_EQ (paper's Qliq_eq)."""
    df = data.copy()
    # Gravity is a fluid property of the string: fill gaps from the same string's other tests.
    bad_gravity = ~df['GRAVITY'].between(10, 60)
    df.loc[bad_gravity, 'GRAVITY'] = np.nan
    df['GRAVITY'] = df['GRAVITY'].fillna(df.groupby('STRING_CODE')['GRAVITY'].transform('median'))

    # Qg = gas lift + formation gas. Without a gas-lift record and without a gas-lift choke the
    # well is naturally flowing, so Qg = formation gas; otherwise leave it missing (imputed).
    no_gas_lift = df['GL_RATE'].isna() & df['GL_CHOKE'].isna()
    df['QG'] = df['TOTAL_GAS'].where(df['TOTAL_GAS'].notna(), df['FM_GAS'].where(no_gas_lift))

    with np.errstate(divide='ignore', invalid='ignore'):
        df['WATER_CUT_FRAC'] = (df['WATER_RATE'] / df['GROSS_FLUID']).clip(0, 1)
    df[TARGET] = df['GROSS_FLUID'] * equivalent_factor(df['GRAVITY'], df['WATER_CUT_FRAC'])
    return df


def data_cleaning(data, exclude_strings=COMMINGLED_STRINGS):
    """Keep tests that follow the domain rules of the paper. Does not impute anything.
    All wells are single string, so the paper's dual-string filter is not needed.

    - real wellhead temperature in °F within a plausible range (no median-filled WHT)
    - valid rates, pressure, choke, gravity and water cut
    - no commingled tests
    """
    df = data[
        (data['GROSS_FLUID'] >= 100)
        & (data['WH_PRESS'] <= 3500)
        & (data['SURF_CHOKE'] > 0) & (data['SURF_CHOKE'] <= 128)
        & data['WH_TEMP'].between(40, 250)
        & data['GRAVITY'].between(10, 60)
        & data['WATER_CUT_FRAC'].between(0, 1)
        & data['TEST_DATE'].notna()
        & ~data['STRING_CODE'].isin(exclude_strings)
    ]
    return df.reset_index(drop=True)


def prepare_data(file_path=DATA_PATH):
    """Load, derive and clean in one step. Returns (raw with derived columns, clean)."""
    data = add_derived_columns(load_data(file_path))
    return data, data_cleaning(data)


def split_data(data, test_size=0.2, method='random', random_state=42):
    """80:20 split. 'random' (default) is the paper's random split. 'time' holds out the latest
    tests of every string (blind data, like predicting the next test of a known well); it is the
    stricter check, because a random split lets the model see a well's neighbouring tests."""
    if method == 'random':
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)
        return train, test
    if method != 'time':
        raise ValueError(f"Unknown split method: {method}")
    df = data.sort_values(['STRING_CODE', 'TEST_DATE'])
    rank = df.groupby('STRING_CODE').cumcount() / df.groupby('STRING_CODE')['STRING_CODE'].transform('size')
    is_test = rank >= 1 - test_size
    return df[~is_test], df[is_test]


def evaluate(y_true_eq, y_pred_eq, factor):
    """Metrics on QLIQ_EQ and on gross liquid in BPD (y / factor)."""
    gross_true, gross_pred = y_true_eq / factor, y_pred_eq / factor
    return {
        'R2_eq': r2_score(y_true_eq, y_pred_eq),
        'R2_gross': r2_score(gross_true, gross_pred),
        'MAE_gross_bpd': mean_absolute_error(gross_true, gross_pred),
        'MedAPE_gross_%': float(np.median(np.abs(gross_pred - gross_true) / gross_true) * 100),
    }


# ---------------------------------------------------------------------------------------------
# Combined regression (model + polynomial on the WHT sensitivity curve)
# ---------------------------------------------------------------------------------------------
def sweep_grid(values, n_points=25):
    """WHT values for the sensitivity sweep: the training range (1st-99th percentile)."""
    lo, hi = np.nanpercentile(values, [1, 99])
    return np.linspace(lo, hi, n_points)


def sensitivity_curves(model, X, grid, sweep_feature=SWEEP_FEATURE):
    """Model prediction for every row of X with sweep_feature set to each grid value.
    Returns an array of shape (len(X), len(grid))."""
    Xr = X.loc[X.index.repeat(len(grid))].copy()
    Xr[sweep_feature] = np.tile(grid, len(X))
    return model.predict(Xr).reshape(len(X), len(grid))


def combined_predict(model, X, grid, degree=POLY_DEGREE, sweep_feature=SWEEP_FEATURE):
    """Paper's combined regression: (1) sweep the model over WHT with the other features fixed,
    (2) fit a polynomial to that curve and evaluate it at the actual WHT. Rows without WHT fall
    back to the plain model prediction."""
    curves = sensitivity_curves(model, X, grid, sweep_feature)
    coef = np.linalg.lstsq(np.vander(grid, degree + 1), curves.T, rcond=None)[0]  # (degree+1, n)
    x = X[sweep_feature].to_numpy(dtype=float)
    pred = sum(coef[k] * x ** (degree - k) for k in range(degree + 1))
    missing = np.isnan(x)
    if missing.any():
        pred[missing] = model.predict(X[missing])
    return pred


# ---------------------------------------------------------------------------------------------
# Random Forest training
# ---------------------------------------------------------------------------------------------
def build_model(random_state=42, **rf_params):
    """Median imputation (fitted on training data only) + Random Forest."""
    params = {**RF_PARAMS, **rf_params}
    return make_pipeline(SimpleImputer(strategy='median'),
                         RandomForestRegressor(random_state=random_state, n_jobs=-1, **params))


def train_model(file_path=DATA_PATH, model_path=MODEL_PATH, split='random', degree=POLY_DEGREE,
                model=None):
    """Train and evaluate the Random Forest, refit on all clean data, save and return the bundle."""
    data, clean = prepare_data(file_path)
    print(f"Clean tests: {len(clean)} of {len(data)} ({clean['WELL_NAME'].nunique()} wells)")

    train, test = split_data(clean, method=split)
    print(f"Split ({split}): train {len(train)}, test {len(test)}")
    model = build_model() if model is None else model
    model.fit(train[FEATURES], train[TARGET])

    grid = sweep_grid(train[SWEEP_FEATURE])
    factor = equivalent_factor(test['GRAVITY'], test['WATER_CUT_FRAC']).to_numpy()
    y_test = test[TARGET].to_numpy()
    metrics = pd.DataFrame({
        'Random Forest': evaluate(y_test, model.predict(test[FEATURES]), factor),
        'Random Forest + polynomial': evaluate(y_test, combined_predict(model, test[FEATURES], grid, degree),
                                               factor),
    }).T
    print(f"\nTest set ({split} split):\n" + metrics.round(3).to_string())

    rf = model[-1]
    if hasattr(rf, 'feature_importances_'):
        importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\nFeature importance:\n" + importance.round(3).to_string())

    # Final model on all clean data, as used in deployment.
    model.fit(clean[FEATURES], clean[TARGET])
    bundle = {'model': model, 'model_name': 'Random Forest', 'features': FEATURES, 'target': TARGET,
              'sweep_feature': SWEEP_FEATURE, 'grid': sweep_grid(clean[SWEEP_FEATURE]), 'degree': degree,
              'test_metrics': metrics.to_dict(orient='index')}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"\nSaved model bundle to {model_path}")
    return bundle


def predict_gross_fluid(bundle, data):
    """Gross liquid (BPD) from a saved bundle. `data` needs FEATURES (incl. GRAVITY, WATER_CUT_FRAC)."""
    X = data[bundle['features']]
    eq = combined_predict(bundle['model'], X, bundle['grid'], bundle['degree'], bundle['sweep_feature'])
    return eq / equivalent_factor(data['GRAVITY'], data['WATER_CUT_FRAC']).to_numpy()


def main():
    train_model()


if __name__ == "__main__":
    main()
