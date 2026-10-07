"""
Virtual Well Test (VWT) - gross fluid rate from surface sensors
================================================================
Field set-up this pipeline assumes
  * No test separator. Gross fluid is measured only when a PORTABLE TEST UNIT visits a well.
  * Between visits, the only live data are surface sensor readings:
        THP (WH_PRESS), CHP (WH_CSG_PRESS), separator/flowline pressure (SEP_PRESS), choke size (SURF_CHOKE)
  * Water cut is NOT a model input. It is applied afterwards (visual sample) to split
    predicted gross into oil and water.
  * Gas-lift data (GL_CHOKE, Qgi) is NOT used - not every well has gas lift.

Model ("anchor + surface tracking")
  Each prediction starts from the last portable-unit rate on that well (the anchor) and
  corrects it for how the surface readings have changed since:
    1. Grey-box choke model : q = q_anchor * (THP/THP_anchor)^a * (S/S_anchor)^b   (a, b fitted field-wide)
    2. ExtraTrees residual  : ML learns log(q/q_anchor) from surface changes + time since anchor
    3. Hybrid               : grey-box for short gaps, ML for long gaps (switch point tuned on an inner per-well split of training data)
  Baselines: persistence (anchor rate carried forward) and a no-anchor surface-only model.

Validation
  Temporal hold-out: last 20% of each well's tests are the "future". Models are trained only on
  earlier tests. Every target is paired with several earlier portable-unit tests so accuracy can
  be reported as a function of DAYS SINCE LAST VISIT -> recalibration-interval design curve.

Outputs (in OUT)
  vwt_horizon_table.csv     MdAPE / within-20% by revisit gap and model
  vwt_horizon_curve.png     accuracy-decay curve
  vwt_no_anchor.csv         reference: what surface data alone can do without any anchor
  vwt_models.joblib         fitted grey-box exponents, ML model, hybrid switch point
  predict_virtual_test()    deployable function (bottom of file)
"""
import warnings, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")

DATA = "/mnt/user-data/uploads/processed_data.csv"
OUT = "/mnt/user-data/outputs/"
SEED = 42
SENSORS = ["WH_PRESS", "WH_CSG_PRESS", "SEP_PRESS", "SURF_CHOKE"]
FULL_OPEN_CODE = 100          # choke codes >= this (117/128/137) treated as full bore
HOLDOUT_FRAC = 0.2
ANCHORS_PER_TARGET = 8
GAP_BINS = [0, 7, 30, 90, 180, 365, 730, 100_000]
GAP_LABELS = ["≤1 wk", "1 wk–1 mo", "1–3 mo", "3–6 mo", "6–12 mo", "1–2 yr", ">2 yr"]
rng = np.random.default_rng(SEED)


# =====================================================================================
# 1. Load, recover well IDs, clean
# =====================================================================================
def recover_well_id(dates: pd.Series) -> np.ndarray:
    """File has several wells stacked with no name column. A new well starts whenever the
    date order flips direction (asc <-> desc); a large jump row joins the new block;
    single orphan rows are merged into the previous block. Replace with real names if available."""
    d = dates.diff().dt.days.fillna(0).values
    seg, s, direc = np.zeros(len(d), int), 0, 0
    for i in range(1, len(d)):
        sg = np.sign(d[i])
        if sg != 0 and direc != 0 and sg != direc:
            s += 1; direc = 0
            if abs(d[i - 1]) > 180 and i >= 2 and seg[i - 2] == s - 1:
                seg[i - 1] = s; direc = sg
        elif sg != 0 and direc == 0:
            direc = sg
        seg[i] = s
    sizes = pd.Series(seg).value_counts()
    for k in sizes[sizes == 1].index:
        if k > 0: seg[seg == k] = k - 1
    return pd.factorize(seg)[0]


def load_clean(path=DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["TEST_DATE"])
    if "WELL_NAME" not in df.columns:
        df["WELL_NAME"] = recover_well_id(df["TEST_DATE"])
    df.loc[df.WH_PRESS > 3500, "WH_PRESS"] = np.nan        # e.g. 5280 psi typos
    df.loc[df.WH_PRESS <= 0, "WH_PRESS"] = np.nan
    df.loc[df.SURF_CHOKE <= 0, "SURF_CHOKE"] = np.nan
    df = (df.sort_values(["WELL_NAME", "TEST_DATE"])
            .drop_duplicates(["WELL_NAME", "TEST_DATE", "WH_PRESS", "SURF_CHOKE"], keep="last"))
    df["WELL_AGE_D"] = (df.TEST_DATE - df.groupby("WELL_NAME").TEST_DATE.transform("min")).dt.days
    # flowing portable-unit tests only; shut-in / dead tests (gross = 0) are not rate targets
    return df[df.GROSS_FLUID > 0].reset_index(drop=True)


# =====================================================================================
# 2. Features
# =====================================================================================
def eff_choke(s):
    return np.clip(s, None, FULL_OPEN_CODE)


def surface_features(cur: pd.DataFrame, prefix="") -> pd.DataFrame:
    """Features from one set of surface readings (no rate information)."""
    thp, chp, sep, s = (cur[c].values for c in SENSORS)
    se = eff_choke(s)
    return pd.DataFrame({
        f"{prefix}THP": thp, f"{prefix}CHP": chp, f"{prefix}SEP": sep, f"{prefix}CHOKE": s,
        f"{prefix}FULL_OPEN": (s >= FULL_OPEN_CODE).astype(float),
        f"{prefix}LOG_GILBERT": np.log1p(thp * se ** 1.89),   # Gilbert choke index
        f"{prefix}DP_CHOKE": thp - sep,                       # drop across choke + flowline
        f"{prefix}P_RATIO": sep / thp,                        # < ~0.55 -> critical flow
        f"{prefix}CHP_THP_DIFF": chp - thp,                   # annulus/tubing head difference
        f"{prefix}CHP_THP_RATIO": chp / thp,
    })


def pair_features(T: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """Features for predicting target test T from anchor (portable-unit) test A."""
    F = surface_features(T)
    F["GAP_D"] = (T.TEST_DATE.values - A.TEST_DATE.values).astype("timedelta64[D]").astype(float)
    F["WELL_AGE_D"] = T.WELL_AGE_D.values
    F["LOG_Q_ANCHOR"] = np.log1p(A.GROSS_FLUID.values)
    for c in SENSORS:
        F[f"ANC_{c}"] = A[c].values
        F[f"D_{c}"] = T[c].values - A[c].values
    F["DLOG_THP"] = np.log(T.WH_PRESS.values) - np.log(A.WH_PRESS.values)
    F["DLOG_CHP"] = np.log(np.clip(T.WH_CSG_PRESS.values, 1, None)) - np.log(np.clip(A.WH_CSG_PRESS.values, 1, None))
    F["DLOG_CHOKE"] = np.log(eff_choke(T.SURF_CHOKE.values)) - np.log(eff_choke(A.SURF_CHOKE.values))
    F["CHOKE_CHANGED"] = (np.abs(F.D_SURF_CHOKE) > 0).astype(float)
    return F.replace([np.inf, -np.inf], np.nan)


def build_pairs(df: pd.DataFrame, is_test: np.ndarray) -> pd.DataFrame:
    """Pair every test with up to N earlier tests on the same well (simulated portable-unit visits).
    Training pairs: target and anchor both in the training period.
    Test pairs: target in hold-out; anchor anywhere earlier (a real measurement is available then)."""
    rows = []
    for _, idx in df.groupby("WELL_NAME").indices.items():
        idx = idx[np.argsort(df.TEST_DATE.values[idx])]
        dates = df.TEST_DATE.values[idx]
        for k in range(1, len(idx)):
            i = idx[k]
            prev = idx[:k][(dates[k] - dates[:k]).astype("timedelta64[D]").astype(int) > 0]
            if not is_test[i]:
                prev = prev[~is_test[prev]]
            if len(prev) == 0:
                continue
            if len(prev) > ANCHORS_PER_TARGET:
                prev = rng.choice(prev, ANCHORS_PER_TARGET, replace=False)
            rows += [(i, j) for j in prev]
    P = pd.DataFrame(rows, columns=["i", "j"])
    P["test"] = is_test[P.i.values]
    return P


# =====================================================================================
# 3. Models
# =====================================================================================
class GreyBoxChoke:
    """log q = log q_anchor + a*dlog(THP) + b*dlog(choke). Field-wide a, b, robust loss."""
    def fit(self, F, y):
        m = F[["DLOG_THP", "DLOG_CHOKE"]].notna().all(axis=1).values
        dthp, ds, base = F.DLOG_THP.values[m], F.DLOG_CHOKE.values[m], F.LOG_Q_ANCHOR.values[m]
        r = least_squares(lambda p: base + p[0] * dthp + p[1] * ds - y[m], [1.0, 1.0],
                          loss="soft_l1", f_scale=0.1)
        self.a, self.b = r.x
        return self

    def predict(self, F):
        corr = self.a * F.DLOG_THP.fillna(0).values + self.b * F.DLOG_CHOKE.fillna(0).values
        return F.LOG_Q_ANCHOR.values + corr


class ResidualML:
    """Learns log(q / q_anchor) from surface changes and time since anchor."""
    def __init__(self, kind="ExtraTrees"):
        self.kind = kind
        if kind == "ExtraTrees":
            self.m = make_pipeline(SimpleImputer(strategy="median"),
                                   ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.7,
                                                       n_jobs=-1, random_state=SEED))
        else:
            self.m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       min_child_samples=15, verbose=-1, random_state=SEED)

    def fit(self, F, y):
        self.cols = list(F.columns)
        self.m.fit(F[self.cols].values, y - F.LOG_Q_ANCHOR.values)
        return self

    def predict(self, F):
        return F.LOG_Q_ANCHOR.values + self.m.predict(F[self.cols].values)


class Hybrid:
    """Grey-box below a gap threshold, ML above it. Threshold chosen on an inner per-well temporal split."""
    def __init__(self, grey, ml, switch_days):
        self.grey, self.ml, self.switch_days = grey, ml, switch_days

    def predict(self, F):
        return np.where(F.GAP_D.values <= self.switch_days, self.grey.predict(F), self.ml.predict(F))


def metrics(y_log, p_log):
    yt, yp = np.expm1(y_log), np.expm1(p_log)
    ape = np.abs(yp - yt) / yt
    return dict(MdAPE=np.median(ape) * 100, within_10=np.mean(ape <= .1) * 100,
                within_20=np.mean(ape <= .2) * 100, MAE=np.mean(np.abs(yp - yt)))


def tune_switch(F, y, train_mask, val_mask):
    """Fit sub-models on the early part of training, choose the gap at which ML takes over."""
    g = GreyBoxChoke().fit(F[train_mask], y[train_mask])
    ml = ResidualML().fit(F[train_mask], y[train_mask])
    Fv, yv = F[val_mask], y[val_mask]
    bins = pd.cut(Fv.GAP_D, GAP_BINS, labels=GAP_LABELS)

    def bin_avg_mdape(t):   # every gap range counts equally, so long gaps don't swamp short ones
        p = Hybrid(g, ml, t).predict(Fv)
        return np.mean([metrics(yv[(bins == b).values], p[(bins == b).values])["MdAPE"]
                        for b in GAP_LABELS if (bins == b).sum() >= 20])
    return min([0, 30, 90, 180, 365, 730, 1e9], key=bin_avg_mdape)


# =====================================================================================
# 4. Run
# =====================================================================================
def main():
    df = load_clean()
    rank = df.groupby("WELL_NAME").TEST_DATE.rank(pct=True)
    is_test = (rank > 1 - HOLDOUT_FRAC).values
    print(f"{len(df)} flowing tests, {df.WELL_NAME.nunique()} wells, {is_test.sum()} hold-out tests")

    P = build_pairs(df, is_test)
    T, A = df.loc[P.i].reset_index(drop=True), df.loc[P.j].reset_index(drop=True)
    F = pair_features(T, A)
    y = np.log1p(T.GROSS_FLUID.values)
    tr, te = ~P.test.values, P.test.values
    print(f"anchor-target pairs: train {tr.sum()}, test {te.sum()}")

    # hybrid switch point: tune on an inner per-well temporal split of the training period
    # (each well's tests between the 60th and 80th percentile act as validation targets)
    inner = (rank > 1 - HOLDOUT_FRAC - 0.2).values & ~is_test
    fit_m = tr & ~inner[P.i.values] & ~inner[P.j.values]
    val_m = tr & inner[P.i.values]
    switch = tune_switch(F, y, fit_m, val_m)

    grey = GreyBoxChoke().fit(F[tr], y[tr])
    et = ResidualML("ExtraTrees").fit(F[tr], y[tr])
    lg = ResidualML("LightGBM").fit(F[tr], y[tr])
    hyb = Hybrid(grey, et, switch)
    print(f"grey-box exponents: THP^{grey.a:.2f}, choke^{grey.b:.2f} (Gilbert 1.00, 1.89); "
          f"hybrid switches to ML after {switch:.0f} days")

    preds = {
        "Persistence (anchor)": F.LOG_Q_ANCHOR.values,
        "Grey-box choke": grey.predict(F),
        "ExtraTrees residual": et.predict(F),
        "LightGBM residual": lg.predict(F),
        "Hybrid": hyb.predict(F),
    }

    # --- accuracy vs days since last portable-unit visit
    gap_bin = pd.cut(F.GAP_D, GAP_BINS, labels=GAP_LABELS)
    rows = []
    for b in GAP_LABELS:
        m = te & (gap_bin == b).values
        if m.sum() < 20:
            continue
        for name, p in preds.items():
            rows.append(dict(gap=b, model=name, n=int(m.sum()), **metrics(y[m], p[m])))
    for name, p in preds.items():
        rows.append(dict(gap="all", model=name, n=int(te.sum()), **metrics(y[te], p[te])))
    H = pd.DataFrame(rows).round(1)
    H.to_csv(OUT + "vwt_horizon_table.csv", index=False)
    print("\nMedian absolute % error by days since last portable-unit visit")
    print(H.pivot(index="gap", columns="model", values="MdAPE")
           .reindex([g for g in GAP_LABELS + ["all"] if g in H.gap.values]).to_string())
    print("\nn per bin:", H.groupby("gap").n.first().to_dict())

    plot_horizon(H)
    no_anchor_reference(df, is_test)

    joblib.dump(dict(grey_a=grey.a, grey_b=grey.b, ml_model=et.m, ml_cols=et.cols, switch_days=switch,
                     sensors=SENSORS, full_open_code=FULL_OPEN_CODE), OUT + "vwt_models.joblib", compress=3)
    print("\nsaved:", OUT + "vwt_models.joblib")


def plot_horizon(H):
    colors = {"Persistence (anchor)": "#8a8a85", "Grey-box choke": "#2a78d6",
              "ExtraTrees residual": "#eb6834", "LightGBM residual": "#1baf7a", "Hybrid": "#111111"}
    D = H[H.gap != "all"]
    order = [g for g in GAP_LABELS if g in D.gap.values]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, c in colors.items():
        s = D[D.model == name].set_index("gap").reindex(order).MdAPE.values
        lw, ms = (2.6, 7) if name == "Hybrid" else (1.8, 5)
        ax.plot(x, s, color=c, lw=lw, marker="o", ms=ms, label=name, zorder=3 if name == "Hybrid" else 2)
    n = D.groupby("gap").n.first().reindex(order).values
    ax.set_xticks(x, [f"{g}\n(n={k})" for g, k in zip(order, n)], fontsize=9)
    ax.set_ylabel("Median absolute error (%)")
    ax.set_xlabel("Time since last portable-unit test")
    ax.set_title("Virtual well test accuracy vs. revisit interval (surface sensors only)", loc="left")
    ax.grid(axis="y", color="#e6e6e3", lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.set_ylim(0, None); ax.set_xlim(-0.3, len(order) - 0.7)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout(); plt.savefig(OUT + "vwt_horizon_curve.png", dpi=140); plt.close()


def no_anchor_reference(df, is_test):
    """What surface readings alone achieve with no portable-unit anchor at prediction time."""
    X = surface_features(df)
    X["WELL_AGE_D"] = df.WELL_AGE_D.values
    X = X.replace([np.inf, -np.inf], np.nan)
    y = np.log1p(df.GROSS_FLUID.values)
    rows = []
    # A: well known from history (well identity as a feature), later tests predicted
    XA = pd.concat([X, pd.get_dummies(df.WELL_NAME, prefix="W").astype(float)], axis=1).values
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, verbose=-1, random_state=SEED)
    m.fit(XA[~is_test], y[~is_test])
    rows.append(dict(case="known well, no recent anchor", **metrics(y[is_test], m.predict(XA[is_test]))))
    # B: never-tested well
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, df.WELL_NAME):
        mm = make_pipeline(SimpleImputer(strategy="median"),
                           ExtraTreesRegressor(400, min_samples_leaf=2, n_jobs=-1, random_state=SEED))
        mm.fit(X.values[a], y[a]); oof[b] = mm.predict(X.values[b])
    rows.append(dict(case="never-tested well", **metrics(y, oof)))
    R = pd.DataFrame(rows).round(1)
    R.to_csv(OUT + "vwt_no_anchor.csv", index=False)
    print("\nReference without any anchor:\n", R.to_string(index=False))


# =====================================================================================
# 5. Deployment
# =====================================================================================
def predict_virtual_test(anchor: dict, current: dict, models=None, water_cut=None) -> dict:
    """Predict gross fluid for one well from its live surface readings.

    anchor   : last portable-unit test on this well
               {TEST_DATE, GROSS_FLUID, WH_PRESS, WH_CSG_PRESS, SEP_PRESS, SURF_CHOKE, WELL_AGE_D}
    current  : current sensor readings {TEST_DATE, WH_PRESS, WH_CSG_PRESS, SEP_PRESS, SURF_CHOKE, WELL_AGE_D}
    water_cut: optional fraction from a visual fluid sample, applied after the gross prediction
    """
    models = models or joblib.load(OUT + "vwt_models.joblib")
    A = pd.DataFrame([anchor]); T = pd.DataFrame([current])
    A["TEST_DATE"] = pd.to_datetime(A.TEST_DATE); T["TEST_DATE"] = pd.to_datetime(T.TEST_DATE)
    F = pair_features(T, A)
    grey = GreyBoxChoke(); grey.a, grey.b = models["grey_a"], models["grey_b"]
    q_grey = float(np.expm1(grey.predict(F))[0])
    q_ml = float(np.expm1(F.LOG_Q_ANCHOR.values + models["ml_model"].predict(F[models["ml_cols"]].values))[0])
    gap = float(F.GAP_D.iloc[0])
    gross = q_grey if gap <= models["switch_days"] else q_ml
    out = dict(gross_bfpd=round(gross, 1), grey_box_bfpd=round(q_grey, 1), ml_bfpd=round(q_ml, 1),
               days_since_anchor=int(gap),
               # large disagreement between the two models -> conditions have drifted; schedule a visit
               retest_flag=bool(abs(np.log(q_grey / q_ml)) > np.log(1.25)))
    if water_cut is not None:
        out["oil_bopd"] = round(gross * (1 - water_cut), 1)
        out["water_bwpd"] = round(gross * water_cut, 1)
    return out


if __name__ == "__main__":
    main()
