# VIRTUAL WELL TEST USING MACHINE LEARNING

## Scope of the Project

A method in providing prediction for well test using machine learning for platform without separator test and depends on the mobile test unit. This method is aimed at providing a solution for well test as well as reducing the cost for mobile test unit.

## Reference

The workflow replicates **IPTC-23662-MS** — Setiadi, Hidayat & Mardin (2024), *Supervised Machine Learning for Liquid Rate Estimation in Oil Field with Limited Well Monitoring Data: A Case Study of the Handil Oil Field* (DOI [10.2523/IPTC-23662-MS](https://doi.org/10.2523/IPTC-23662-MS)), with the data available here:

| Paper | This project |
|---|---|
| Inputs: WHFP, WHFT, Qg, water cut, est. BHT (from reservoir depth) | Same, **without BHT** (no depth data), plus choke, casing pressure, separator pressure and oil gravity |
| Target: mass-equivalent liquid rate `Qliq_eq` | Same (`QLIQ_EQ`), converted back to gross BPD for reporting |
| Remove dual-string and commingled tests | Commingled only (all wells are single string) |
| Random 80:20 split | Same by default; a time split is available as a stricter check |
| Linear, polynomial, SVM, random forest, ANN | Same, plus XGBoost |
| Random forest + polynomial on the WHT sensitivity sweep | Same |

## Workflow

1. **Load** well tests (`data/welltest.csv` or `.xlsx`).
2. **Derive** the paper's variables:
   - `QG` — gas lift injection + formation gas (MSCFPD); formation gas only for naturally flowing wells
   - `WATER_CUT_FRAC` — `WATER_RATE / GROSS_FLUID`
   - `QLIQ_EQ` — `OIL_RATE × SG_oil + WATER_RATE × 1.0`, with `SG_oil = 141.5 / (131.5 + API)`
   - invalid oil gravity is filled from the same string's other tests
3. **Clean**:
   - only tests with a real wellhead temperature (40–250 °F); missing WHT is dropped, not imputed
   - `GROSS_FLUID ≥ 100`, `WH_PRESS ≤ 3500`, `0 < SURF_CHOKE ≤ 128`, API 10–60
   - commingled strings listed in `COMMINGLED_STRINGS` are removed (empty by default; fill from operations knowledge)
4. **Split** 80:20 at random (paper) or by time (`split='time'`: each string's latest 20% of tests).
5. **Train** a Random Forest (median imputation fitted on the training set only).
6. **Combined regression**: sweep the model over WHT with the other features fixed, fit a degree-2 polynomial to the curve, evaluate it at the actual WHT. This keeps a trend outside the training range instead of a flat prediction.
7. **Refit** on all clean data and save the model bundle to `model/rf_model.pkl`.

## Scripts

| File | Purpose |
|---|---|
| `src/training01.py` | Training script: data preparation (shared), Random Forest training, evaluation and saving. Hyper-parameters in `RF_PARAMS`. |
| `src/test01.py` | Model selection and analysis: Spearman correlation, tuned comparison of all algorithms, permutation feature importance, leave-wells-out check, plots in `Images/`. Does not save a model. |
| `tests/test_training01.py` | Unit tests for both scripts (synthetic data, no writes to `model/` or `Images/`). |

### Usage

```bash
cd src
python training01.py        # train and save model/rf_model.pkl
python test01.py            # compare algorithms and write plots to Images/
cd .. && python -m unittest discover -s tests
```

Both entry points take a split option, e.g. `training01.train_model(split='time')` or `test01.main(split='time')`.

When `test01.py` selects different Random Forest hyper-parameters, it prints them; copy them into `training01.RF_PARAMS`. Model selection uses time-block cross-validation within the training set so that models which memorise a well's neighbouring tests are not favoured.

### Predicting with the saved model

```python
import joblib
import training01

bundle = joblib.load('model/rf_model.pkl')
# df needs training01.FEATURES:
# WH_PRESS, WH_TEMP, QG, WATER_CUT_FRAC, SURF_CHOKE, WH_CSG_PRESS, SEP_PRESS, GRAVITY
gross_bpd = training01.predict_gross_fluid(bundle, df)
```

At deployment `QG` is not measured; estimate it as gas lift rate + GOR (last test) × oil estimate, as in the paper. Water cut comes from fluid sampling.

## Results

980 clean tests from 22 wells.

| Model | R² (QLIQ_EQ) | R² (gross BPD) | Median abs. error |
|---|---|---|---|
| **Random Forest** (random split) | **0.90** | 0.89 | 5.4 % |
| Random Forest + polynomial | 0.89 | 0.88 | 7.1 % |
| Paper (Handil, with BHT) | 0.93 / 0.91 combined | – | – |

Algorithm ranking matches the paper: Random Forest > XGBoost > SVM > ANN > polynomial > linear. Unlike the paper, choke size and oil gravity are the dominant features and WHT contributes little, so the polynomial step changes the result very little.

### Interpreting the score

Wells are tested very often (82% of test rows have a training test of the same string within 1 day), so a random split lets the model recognise a well from its neighbouring tests. The same model on stricter splits:

| Split | R² |
|---|---|
| Random (paper) | 0.90 |
| Whole string-months held out | 0.78 |
| Each string's latest 20% of tests (`split='time'`) | 0.55 |
| Wells never seen in training | negative |

Report the random-split score for comparison with the paper and the stricter scores as the realistic accuracy for predicting future tests. On the time split, simply using the last test of the same well (R² 0.88) is still better than the model when a recent test exists.

## Limitations

- No BHT / reservoir depth, the second most important feature in the paper.
- The model works for wells present in the training history; it does not generalise to new wells.
- Commingled tests are not flagged in the data; fill `COMMINGLED_STRINGS` to exclude them.
- WHT is only recorded from about 2010, which limits the clean dataset to ~25% of all tests.

## Requirements

Python 3, pandas, numpy, scikit-learn, joblib, matplotlib; xgboost is optional (used by `test01.py` if installed).
