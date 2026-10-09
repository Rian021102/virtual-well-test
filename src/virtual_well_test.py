"""
Virtual Well Test (VWT) - gross fluid rate prediction
=====================================================
Pipeline:
  1. Recover a proxy WELL_ID (file has several wells stacked, asc/desc date order)
  2. Clean obvious outliers / typos
  3. Feature engineering (choke physics, gas-lift, pressure ratios, last-test memory)
  4. Benchmark several models under two validation schemes
       A) Temporal hold-out per well  -> "VWT on known wells" (the real use case)
       B) Leave-wells-out GroupKFold  -> "VWT on a well with no test history"
Target: GROSS_FLUID (bfpd) on flowing tests (GROSS_FLUID > 0), modelled in log space.
OIL_RATE, WATER_RATE, GRAVITY are excluded as inputs (measured *by* the test -> leakage).
"""
import warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# import lightgbm as lgb, xgboost as xgb
warnings.filterwarnings("ignore")
OUT = "/home/rianr/pypro/myvenv/virtual-well-test/outputs"
SEED = 42

# ---------------------------------------------------------------- 1. load + well id
df = pd.read_csv("/home/rianr/pypro/myvenv/virtual-well-test/data/processed_data.csv", parse_dates=["TEST_DATE"])

def recover_well_id(dates):
    """New well whenever the date-order direction flips; singletons merged back."""
    d = dates.diff().dt.days.fillna(0).values
    seg, s, direc = np.zeros(len(d), int), 0, 0
    for i in range(1, len(d)):
        sg = np.sign(d[i])
        if sg != 0 and direc != 0 and sg != direc:
            s += 1; direc = 0
            if abs(d[i-1]) > 180 and i >= 2 and seg[i-2] == s-1:   # jump row starts new well
                seg[i-1] = s; direc = sg
        elif sg != 0 and direc == 0:
            direc = sg
        seg[i] = s
    sizes = pd.Series(seg).value_counts()
    for k in sizes[sizes == 1].index:                              # orphan rows -> previous well
        if k > 0: seg[seg == k] = k - 1
    return pd.factorize(seg)[0]

df["WELL_ID"] = recover_well_id(df["TEST_DATE"])

# ---------------------------------------------------------------- 2. cleaning
df.loc[df.WH_PRESS > 3500, "WH_PRESS"] = np.nan          # e.g. 5280 psi typos
df.loc[df.WH_TEMP > 250, "WH_TEMP"] = np.nan             # 889 degF typo
df.loc[df.GL_CHOKE > 200, "GL_CHOKE"] = np.nan           # 487, 2260 -> not 64ths
df.loc[df.SURF_CHOKE <= 0, "SURF_CHOKE"] = np.nan
df = df.sort_values(["WELL_ID", "TEST_DATE"]).drop_duplicates(
        ["WELL_ID", "TEST_DATE", "WH_PRESS", "SURF_CHOKE"], keep="last").reset_index(drop=True)

# ---------------------------------------------------------------- 3. features
g = df.groupby("WELL_ID")
df["WC"] = np.where(df.GROSS_FLUID > 0, df.WATER_RATE / df.GROSS_FLUID.replace(0, np.nan), np.nan)

# choke physics (choke in 1/64 in). >=100 codes behave like "full bore / no choke"
df["CHOKE_FULL_OPEN"] = (df.SURF_CHOKE >= 100).astype(float)
df["CHOKE_AREA"] = (df.SURF_CHOKE.clip(upper=100) / 64.0) ** 2
df["GILBERT_IDX"] = df.WH_PRESS * df.SURF_CHOKE.clip(upper=100) ** 1.89     # q ~ P*S^1.89 / (C*GLR^0.546)
df["LOG_GILBERT"] = np.log1p(df.GILBERT_IDX)
df["DP_CHOKE"] = df.WH_PRESS - df.SEP_PRESS
df["P_RATIO"] = df.SEP_PRESS / df.WH_PRESS                                   # <~0.55 critical flow
df["CHP_THP_DIFF"] = df.WH_CSG_PRESS - df.WH_PRESS                           # gas-lift driving head
df["CHP_THP_RATIO"] = df.WH_CSG_PRESS / df.WH_PRESS
df["GL_ACTIVE"] = (df.GL_CHOKE.fillna(0) > 0).astype(float)
df["GL_INJ_PROXY"] = df.WH_CSG_PRESS * (df.GL_CHOKE.fillna(0) / 64.0) ** 2   # orifice-like injection proxy

# well life / time
df["WELL_AGE_D"] = (df.TEST_DATE - g.TEST_DATE.transform("min")).dt.days
df["YEAR"] = df.TEST_DATE.dt.year

# memory of the last *valid flowing* test (known before the virtual test is needed)
flow = df.GROSS_FLUID.where(df.GROSS_FLUID > 0)
df["_last_gf"] = flow; df["_last_wc"] = df.WC.where(df.GROSS_FLUID > 0)
df["_last_gil"] = df.GILBERT_IDX.where(df.GROSS_FLUID > 0)
df["_last_date"] = df.TEST_DATE.where(df.GROSS_FLUID > 0)
for c in ["_last_gf", "_last_wc", "_last_gil", "_last_date"]:
    df[c] = g[c].transform(lambda s: s.shift(1).ffill())
df["PREV_GF"] = df._last_gf
df["PREV_WC"] = df._last_wc
df["LOG_PREV_GF"] = np.log1p(df.PREV_GF)
df["DAYS_SINCE_TEST"] = (df.TEST_DATE - df._last_date).dt.days
df["GILBERT_SCALED_PREV"] = df.PREV_GF * df.GILBERT_IDX / df._last_gil      # choke/THP-corrected last rate
df["LOG_GILBERT_SCALED_PREV"] = np.log1p(df.GILBERT_SCALED_PREV)
for c in ["WH_PRESS", "WH_CSG_PRESS", "SURF_CHOKE"]:
    df[f"D_{c}"] = g[c].diff()                                              # change vs previous test
df["ROLL3_GF"] = g["_last_gf"].transform(lambda s: s.rolling(3, min_periods=1).mean())
df = df.replace([np.inf, -np.inf], np.nan)

RAW = ["WH_CSG_PRESS", "WH_PRESS", "WH_TEMP", "SURF_CHOKE", "GL_CHOKE", "SEP_PRESS", "DURATION"]
PHYS = ["CHOKE_FULL_OPEN", "CHOKE_AREA", "LOG_GILBERT", "DP_CHOKE", "P_RATIO", "CHP_THP_DIFF",
        "CHP_THP_RATIO", "GL_ACTIVE", "GL_INJ_PROXY", "WELL_AGE_D", "YEAR"]
MEM = ["LOG_PREV_GF", "PREV_WC", "DAYS_SINCE_TEST", "LOG_GILBERT_SCALED_PREV", "ROLL3_GF",
       "D_WH_PRESS", "D_WH_CSG_PRESS", "D_SURF_CHOKE"]
FEATURE_SETS = {"raw": RAW, "raw+physics": RAW + PHYS, "raw+physics+memory": RAW + PHYS + MEM}

data = df[(df.GROSS_FLUID > 0)].copy()
y = np.log1p(data.GROSS_FLUID.values)

# ---------------------------------------------------------------- 4. models
def models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Ridge (log)": make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "RandomForest": make_pipeline(imp(), RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=SEED)),
        "ExtraTrees": make_pipeline(imp(), ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.7, n_jobs=-1, random_state=SEED)),
        "HistGB": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED),
        # "LightGBM": lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
        #                               colsample_bytree=0.8, min_child_samples=15, reg_lambda=1.0, verbose=-1, random_state=SEED),
        "XGBoost": xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                    min_child_weight=3, reg_lambda=1.0, random_state=SEED),
    }

def score(yt_log, yp_log):
    yt, yp = np.expm1(yt_log), np.expm1(yp_log)
    ape = np.abs(yp - yt) / yt
    return dict(MAE=mean_absolute_error(yt, yp), R2=r2_score(yt, yp), MdAPE=np.median(ape) * 100,
                within_10=np.mean(ape <= .10) * 100, within_20=np.mean(ape <= .20) * 100)

# Scheme A: last 20% of each well's flowing tests held out (temporal)
rank = data.groupby("WELL_ID").TEST_DATE.rank(pct=True)
trA, teA = (rank <= 0.8).values, (rank > 0.8).values
# Scheme B: leave-wells-out
gkf = GroupKFold(n_splits=5)

rows, preds_A = [], {}
# naive baselines (scheme A)
for name, col in [("Baseline: last test", "PREV_GF"), ("Baseline: Gilbert-scaled last test", "GILBERT_SCALED_PREV")]:
    m = teA & data[col].notna().values & (data[col] > 0).values
    rows.append(dict(scheme="A temporal", features="-", model=name, **score(y[m], np.log1p(data[col].values[m]))))

for fs_name, feats in FEATURE_SETS.items():
    X = data[feats].values
    for mname, model in models().items():
        model.fit(X[trA], y[trA]); p = model.predict(X[teA])
        rows.append(dict(scheme="A temporal", features=fs_name, model=mname, **score(y[teA], p)))
        if fs_name == "raw+physics+memory": preds_A[mname] = p
        # scheme B uses only non-memory features for a no-history well; memory set also run for reference
        oof = np.zeros(len(y))
        for tr, te in gkf.split(X, y, data.WELL_ID):
            mb = models()[mname]; mb.fit(X[tr], y[tr]); oof[te] = mb.predict(X[te])
        rows.append(dict(scheme="B leave-wells-out", features=fs_name, model=mname, **score(y, oof)))
    print("done", fs_name)

res = pd.DataFrame(rows).round(2)
res.to_csv(OUT + "vwt_model_benchmark.csv", index=False)
print(res.sort_values(["scheme", "MdAPE"]).to_string(index=False))

# ---------------------------------------------------------------- 5. plots for best model
best = res[(res.scheme == "A temporal") & (res.features == "raw+physics+memory")].sort_values("MdAPE").iloc[0].model
X = data[RAW + PHYS + MEM]
fm = models()[best]; fm.fit(X.values[trA], y[trA])
est = fm[-1] if hasattr(fm, "steps") else fm
imp_ = pd.Series(getattr(est, "feature_importances_", np.zeros(X.shape[1])), index=X.columns)
if imp_.sum() == 0:
    from sklearn.inspection import permutation_importance
    imp_ = pd.Series(permutation_importance(fm, X.values[teA], y[teA], n_repeats=5, random_state=SEED).importances_mean, index=X.columns)
imp_ = (imp_ / imp_.sum()).sort_values()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
yt, yp = np.expm1(y[teA]), np.expm1(preds_A[best])
ax[0].scatter(yt, yp, s=10, alpha=.5)
lim = [1, max(yt.max(), yp.max()) * 1.1]
ax[0].plot(lim, lim, "k-"); ax[0].plot(lim, [l * 1.2 for l in lim], "r--", lw=.8); ax[0].plot(lim, [l * .8 for l in lim], "r--", lw=.8)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("Measured gross fluid (bfpd)"); ax[0].set_ylabel("Virtual test (bfpd)")
ax[0].set_title(f"{best} - temporal hold-out (±20% band)")
imp_.tail(18).plot.barh(ax=ax[1]); ax[1].set_title(f"{best} feature importance (normalised)")
plt.tight_layout(); plt.savefig(OUT + "vwt_best_model.png", dpi=130)

# per-well error, temporal hold-out
data_te = data[teA].assign(pred=yp)
pw = data_te.groupby("WELL_ID").apply(lambda d: pd.Series(dict(n=len(d),
        MdAPE=np.median(np.abs(d.pred - d.GROSS_FLUID) / d.GROSS_FLUID) * 100))).round(1)
pw.to_csv(OUT + "vwt_per_well_error.csv")
print("\nbest:", best); print(pw.to_string())
print("\nimportance:\n", imp_.sort_values(ascending=False).round(3).to_string())

# ---------------------------------------------------------------- 6. residual-on-last-test + regime breakdown
# The honest benchmark for a VWT is "carry the last test forward". Model the *change* instead,
# and score separately on tests where operating conditions changed (choke, >10% THP, >30 d gap).
ok = data.PREV_GF.notna().values
base = np.log1p(data.PREV_GF.values)
y_res = y - base
Xf = data[RAW + PHYS + MEM].values
changed = ((data.D_SURF_CHOKE.abs() > 0) | (data.D_WH_PRESS.abs() / data.WH_PRESS > 0.1)
           | (data.DAYS_SINCE_TEST > 30)).values
rows2 = []
for lab, mask in [("all", teA & ok), ("changed", teA & ok & changed), ("steady", teA & ok & ~changed)]:
    rows2.append(dict(subset=lab, model="Persistence (last test)", n=mask.sum(), **score(y[mask], base[mask])))
for mname in ["RandomForest", "ExtraTrees", "LightGBM", "Ridge (log)"]:
    m = models()[mname]; m.fit(Xf[trA & ok], y_res[trA & ok]); p = base + m.predict(Xf)
    for lab, mask in [("all", teA & ok), ("changed", teA & ok & changed), ("steady", teA & ok & ~changed)]:
        rows2.append(dict(subset=lab, model=f"{mname} residual", n=mask.sum(), **score(y[mask], p[mask])))
res2 = pd.DataFrame(rows2).round(2).sort_values(["subset", "MdAPE"])
res2.to_csv(OUT + "vwt_residual_vs_persistence.csv", index=False)
print(res2.to_string(index=False))
