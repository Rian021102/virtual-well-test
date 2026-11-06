"""
Model selection and analysis for the liquid-rate model (IPTC-23662-MS workflow, no BHT).

Uses the data preparation from training01.py so both scripts see exactly the same clean data,
then:
  - Spearman correlation of features vs target (paper figure 10)
  - tunes and compares linear, polynomial, SVM, random forest, ANN and XGBoost (paper table 1)
  - permutation feature importance of the selected model (paper figure 11)
  - combined regression (model + polynomial on the WHT sweep) on the blind test (figures 12-13)
  - leave-wells-out check: how the model does on wells it has never seen
  - plots in Images/

Nothing is saved to model/; train the production model with training01.py. When this script
selects different Random Forest hyper-parameters, copy them into training01.RF_PARAMS.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from training01 import (DATA_PATH, FEATURES, POLY_DEGREE, ROOT, SWEEP_FEATURE, TARGET,
                        combined_predict, equivalent_factor, evaluate, prepare_data,
                        sensitivity_curves, split_data, sweep_grid)

try:
    import xgboost as xgb
except ImportError:  # optional
    xgb = None

IMAGES_DIR = ROOT / 'Images'


def spearman_correlation(data, features=FEATURES, target=TARGET):
    return data[features + [target]].corr(method='spearman')


def candidate_models(random_state=42):
    """Algorithms tested in the paper (plus XGBoost) with their hyper-parameter grids."""
    imp = lambda: SimpleImputer(strategy='median')  # noqa: E731
    scaled_y = lambda est: TransformedTargetRegressor(est, transformer=StandardScaler())  # noqa: E731
    models = {
        'Linear': (make_pipeline(imp(), StandardScaler(), LinearRegression()), {}),
        'Polynomial': (make_pipeline(imp(), StandardScaler(), PolynomialFeatures(), Ridge()),
                       {'polynomialfeatures__degree': [2, 3], 'ridge__alpha': [1, 10, 100]}),
        'SVM': (make_pipeline(imp(), StandardScaler(), scaled_y(SVR())),
                {'transformedtargetregressor__regressor__C': [1, 10, 100],
                 'transformedtargetregressor__regressor__gamma': ['scale', 0.05]}),
        'Random Forest': (make_pipeline(imp(), RandomForestRegressor(n_estimators=300, random_state=random_state,
                                                                     n_jobs=-1)),
                          {'randomforestregressor__min_samples_leaf': [1, 2, 5],
                           'randomforestregressor__max_features': [1.0, 0.5]}),
        'ANN': (make_pipeline(imp(), StandardScaler(),
                              scaled_y(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000,
                                                    early_stopping=True, random_state=random_state))),
                {'transformedtargetregressor__regressor__alpha': [1e-3, 1e-1, 1.0]}),
    }
    if xgb is not None:
        models['XGBoost'] = (make_pipeline(imp(), xgb.XGBRegressor(n_estimators=300, random_state=random_state,
                                                                   n_jobs=-1)),
                             {'xgbregressor__max_depth': [3, 5],
                              'xgbregressor__learning_rate': [0.03, 0.1]})
    return models


def time_block_folds(train, cv=5):
    """CV folds made of time blocks within each string: fold k holds the k-th chronological
    fifth of every string's tests. Unlike shuffled k-fold, neighbouring tests of the same well
    are not split across train/validation, so models that memorise wells are not rewarded."""
    df = train.reset_index(drop=True)
    order = df.sort_values(['STRING_CODE', 'TEST_DATE']).index
    rank = df.loc[order].groupby('STRING_CODE').cumcount() / df.loc[order].groupby('STRING_CODE')[
        'STRING_CODE'].transform('size')
    block = np.empty(len(df), dtype=int)
    block[order] = np.minimum((rank * cv).astype(int), cv - 1)
    return [(np.flatnonzero(block != k), np.flatnonzero(block == k)) for k in range(cv)]


def compare_models(train, test, features=FEATURES, models=None, cv=5, random_state=42):
    """Tune each algorithm with time-block CV on the training set and score it on the blind test.
    Returns (results sorted by CV R2, fitted best estimators, best params)."""
    models = candidate_models(random_state) if models is None else models
    factor = equivalent_factor(test['GRAVITY'], test['WATER_CUT_FRAC']).to_numpy()
    train = train.reset_index(drop=True)
    folds = time_block_folds(train, cv)
    rows, fitted, params = [], {}, {}
    for name, (est, grid) in models.items():
        search = GridSearchCV(est, grid or {}, cv=folds, scoring='r2', n_jobs=-1)
        search.fit(train[features], train[TARGET])
        fitted[name], params[name] = search.best_estimator_, search.best_params_
        pred = search.best_estimator_.predict(test[features])
        rows.append({'model': name, 'cv_R2': search.best_score_,
                     **evaluate(test[TARGET].to_numpy(), pred, factor)})
    results = pd.DataFrame(rows).set_index('model').sort_values('cv_R2', ascending=False)
    return results, fitted, params


def feature_importance(model, X, y, random_state=42):
    """Permutation importance (works for every algorithm), normalised to sum to 1."""
    imp = permutation_importance(model, X, y, n_repeats=10, random_state=random_state, n_jobs=-1)
    vals = np.clip(imp.importances_mean, 0, None)
    vals = vals / vals.sum() if vals.sum() > 0 else vals
    return pd.Series(vals, index=X.columns).sort_values(ascending=False)


def new_well_check(model, data, features=FEATURES, n_splits=5):
    """Leave-wells-out CV R2: how the model does on wells it has never seen."""
    n_splits = min(n_splits, data['WELL_NAME'].nunique())
    return cross_val_score(model, data[features], data[TARGET], groups=data['WELL_NAME'],
                           cv=GroupKFold(n_splits), scoring='r2')


def save_plots(corr, importance, model, train, test, pred_test, grid, out_dir=IMAGES_DIR,
               degree=POLY_DEGREE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title('Spearman correlation')
    fig.tight_layout(); fig.savefig(out_dir / 'spearman_heatmap.png', dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    importance[::-1].plot.barh(ax=ax)
    ax.set_xlabel('Normalised permutation importance'); ax.set_title('Feature importance')
    fig.tight_layout(); fig.savefig(out_dir / 'feature_importance.png', dpi=120); plt.close(fig)

    # Sensitivity on WHT at the median of the other features, extended past the training range.
    base = train[FEATURES].median().to_frame().T
    span = grid[-1] - grid[0]
    wide = np.linspace(grid[0] - 0.2 * span, grid[-1] + 0.2 * span, 60)
    raw = sensitivity_curves(model, base, wide)[0]
    fitted = np.polyval(np.polyfit(grid, sensitivity_curves(model, base, grid)[0], degree), wide)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wide, raw, label='Model only')
    ax.plot(wide, fitted, label=f'Combined (polynomial deg {degree})')
    ax.axvspan(grid[0], grid[-1], alpha=0.1, label='Training range')
    ax.set_xlabel('WH_TEMP (°F)'); ax.set_ylabel('QLIQ_EQ'); ax.set_title('WHT sensitivity'); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / 'wht_sensitivity.png', dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(test[TARGET], pred_test, s=12, alpha=0.6)
    lim = [0, max(test[TARGET].max(), pred_test.max()) * 1.05]
    ax.plot(lim, lim, 'r--'); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual QLIQ_EQ'); ax.set_ylabel('Predicted QLIQ_EQ'); ax.set_title('Blind test')
    fig.tight_layout(); fig.savefig(out_dir / 'actual_vs_predicted.png', dpi=120); plt.close(fig)


def main(file_path=DATA_PATH, split='random', degree=POLY_DEGREE, make_plots=True, models=None,
         out_dir=IMAGES_DIR):
    data, clean = prepare_data(file_path)
    print(f"Clean tests: {len(clean)} of {len(data)} ({clean['WELL_NAME'].nunique()} wells)")

    corr = spearman_correlation(clean)
    print("\nSpearman correlation with QLIQ_EQ:\n" + corr[TARGET].drop(TARGET).round(2).to_string())

    train, test = split_data(clean, method=split)
    print(f"\nSplit ({split}): train {len(train)}, test {len(test)}")
    results, fitted, params = compare_models(train, test, models=models)

    best_name = results.index[0]
    best = fitted[best_name]
    grid = sweep_grid(train[SWEEP_FEATURE])
    factor = equivalent_factor(test['GRAVITY'], test['WATER_CUT_FRAC']).to_numpy()
    combined = evaluate(test[TARGET].to_numpy(), combined_predict(best, test[FEATURES], grid, degree), factor)
    results.loc[f'{best_name} + polynomial', list(combined)] = list(combined.values())
    print("\nModel comparison (cv_R2 = time-block CV on train; others on test set):")
    print(results.round(3).to_string())
    print(f"\nSelected: {best_name}  params: {params[best_name]}")
    if 'Random Forest' in params:
        print(f"Random Forest params (for training01.RF_PARAMS): {params['Random Forest']}")

    importance = feature_importance(best, test[FEATURES], test[TARGET])
    print("\nFeature importance:\n" + importance.round(3).to_string())

    new_well = new_well_check(best, clean)
    print(f"\nLeave-wells-out R2 (unseen wells): mean {new_well.mean():.2f}, per fold {np.round(new_well, 2)}")

    if make_plots:
        save_plots(corr, importance, best, train, test, best.predict(test[FEATURES]), grid,
                   out_dir=out_dir, degree=degree)
    return results, fitted, params


if __name__ == "__main__":
    main()
