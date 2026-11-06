import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.environ.get('TRAINING01_SRC',
                                  os.path.join(os.path.dirname(__file__), '..', 'src')))
import training01  # noqa: E402
import test01  # noqa: E402

FEATURES = training01.FEATURES
TARGET = training01.TARGET


def make_raw_data(n_wells=6, tests_per_well=20, seed=0):
    """Synthetic single-string well tests with the raw columns load_data expects."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        dates = pd.date_range('2015-01-01', periods=tests_per_well, freq='60D')
        gross = rng.uniform(300, 4000, tests_per_well)
        wc = rng.uniform(0, 0.9, tests_per_well)
        for i in range(tests_per_well):
            rows.append({
                'WELL_NAME': f'W-{w}', 'STRING_CODE': f'W-{w}_S', 'TEST_DATE': dates[i].strftime('%Y-%m-%d'),
                'WH_CSG_PRESS': rng.uniform(100, 1500), 'WH_PRESS': rng.uniform(100, 3000),
                'WH_TEMP': 70 + gross[i] / 100, 'GRAVITY': 30.0 + w,
                'OIL_RATE': gross[i] * (1 - wc[i]), 'WATER_RATE': gross[i] * wc[i], 'GROSS_FLUID': gross[i],
                'SURF_CHOKE': rng.uniform(8, 128), 'SEP_PRESS': rng.uniform(50, 300),
                'GL_RATE': rng.uniform(0, 1000), 'GL_CHOKE': 20.0, 'FM_GAS': rng.uniform(0, 2000),
                'TOTAL_GAS': np.nan, 'EXTRA': 1,
            })
    df = pd.DataFrame(rows)
    df['TOTAL_GAS'] = df['GL_RATE'] + df['FM_GAS']
    return df


def prepared(raw=None):
    raw = make_raw_data() if raw is None else raw
    raw = raw.drop(columns='EXTRA').copy()
    raw['TEST_DATE'] = pd.to_datetime(raw['TEST_DATE'])
    return training01.add_derived_columns(raw)


def fast_models():
    return {
        'Linear': (make_pipeline(SimpleImputer(strategy='median'), LinearRegression()), {}),
        'Random Forest': (make_pipeline(SimpleImputer(strategy='median'),
                                        RandomForestRegressor(n_estimators=20, random_state=0)), {}),
    }


class TestLoadData(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.raw = make_raw_data()
        self.csv_path = os.path.join(self.tmpdir.name, 'welltest.csv')
        self.raw.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_keeps_required_columns_and_parses_dates(self):
        data = training01.load_data(self.csv_path)
        self.assertEqual(list(data.columns), training01.RAW_COLS)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['TEST_DATE']))
        self.assertEqual(len(data), len(self.raw))

    def test_xlsx_uses_read_excel(self):
        with mock.patch.object(training01.pd, 'read_excel', return_value=self.raw) as read_excel:
            data = training01.load_data('welltest.xlsx')
        read_excel.assert_called_once_with('welltest.xlsx')
        self.assertEqual(len(data), len(self.raw))

    def test_missing_required_column_raises(self):
        path = os.path.join(self.tmpdir.name, 'bad.csv')
        self.raw.drop(columns='SEP_PRESS').to_csv(path, index=False)
        with self.assertRaisesRegex(KeyError, 'SEP_PRESS'):
            training01.load_data(path)

    def test_unsupported_extension_raises(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported file type'):
            training01.load_data('welltest.txt')


class TestDerivedColumns(unittest.TestCase):

    def base(self, **kw):
        row = {'STRING_CODE': 'A', 'GRAVITY': 35.0, 'GROSS_FLUID': 1000.0, 'WATER_RATE': 400.0,
               'GL_RATE': 300.0, 'GL_CHOKE': 20.0, 'FM_GAS': 500.0, 'TOTAL_GAS': 800.0}
        row.update(kw)
        return pd.DataFrame([row])

    def test_equivalent_liquid_rate(self):
        df = training01.add_derived_columns(self.base())
        sg = 141.5 / (131.5 + 35)
        self.assertAlmostEqual(df.loc[0, 'WATER_CUT_FRAC'], 0.4)
        self.assertAlmostEqual(df.loc[0, TARGET], 600 * sg + 400 * 1.0)

    def test_gas_rate_uses_total_gas(self):
        self.assertEqual(training01.add_derived_columns(self.base()).loc[0, 'QG'], 800)

    def test_gas_rate_falls_back_to_formation_gas_for_natural_flow(self):
        df = training01.add_derived_columns(self.base(TOTAL_GAS=np.nan, GL_RATE=np.nan, GL_CHOKE=np.nan))
        self.assertEqual(df.loc[0, 'QG'], 500)

    def test_gas_rate_unknown_when_gas_lift_not_recorded(self):
        df = training01.add_derived_columns(self.base(TOTAL_GAS=np.nan, GL_RATE=np.nan))
        self.assertTrue(np.isnan(df.loc[0, 'QG']))

    def test_bad_gravity_filled_from_same_string(self):
        df = pd.concat([self.base(), self.base(GRAVITY=534.6), self.base(STRING_CODE='B', GRAVITY=20.0)],
                       ignore_index=True)
        out = training01.add_derived_columns(df)
        self.assertEqual(list(out['GRAVITY']), [35.0, 35.0, 20.0])

    def test_does_not_modify_input(self):
        df = self.base(GRAVITY=534.6)
        original = df.copy()
        training01.add_derived_columns(df)
        pd.testing.assert_frame_equal(df, original)


class TestDataCleaning(unittest.TestCase):

    def test_filters_out_of_range_rows(self):
        df = prepared(make_raw_data(n_wells=1, tests_per_well=7))
        df.loc[1, 'WH_PRESS'] = 3501
        df.loc[2, 'GROSS_FLUID'] = 99
        df.loc[3, 'SURF_CHOKE'] = 129
        df.loc[4, 'WH_TEMP'] = np.nan        # missing WHT is dropped, not imputed
        df.loc[5, 'WH_TEMP'] = 889
        cleaned = training01.data_cleaning(df)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(list(cleaned.index), [0, 1])
        self.assertFalse(cleaned['WH_TEMP'].isna().any())

    def test_excludes_commingled_strings(self):
        df = prepared()
        cleaned = training01.data_cleaning(df, exclude_strings=['W-0_S'])
        self.assertNotIn('W-0_S', set(cleaned['STRING_CODE']))
        self.assertEqual(len(cleaned), len(df) - 20)

    def test_does_not_impute(self):
        df = prepared()
        df.loc[0, 'SEP_PRESS'] = np.nan
        self.assertTrue(training01.data_cleaning(df)['SEP_PRESS'].isna().any())


class TestDualString(unittest.TestCase):

    def frame(self, rows):
        df = pd.DataFrame(rows, columns=['WELL_NAME', 'STRING_CODE', 'TEST_DATE'])
        df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE'])
        return df

    def test_drops_tests_while_other_string_active(self):
        df = self.frame([
            ['A', 'A_L', '2010-01-01'], ['A', 'A_S', '2010-02-01'],   # both active -> drop both
            ['A', 'A_S', '2014-01-01'],                                # L inactive by then -> keep
            ['B', 'B_S', '2010-01-01'],                                # single string -> keep
        ])
        kept = training01.remove_dual_string_tests(df)
        self.assertEqual(list(kept.index), [2, 3])


class TestSplit(unittest.TestCase):

    def test_time_split_holds_out_latest_tests_of_each_string(self):
        df = prepared()
        train, test = training01.split_data(df, method='time')
        self.assertEqual(len(test), 6 * 4)
        self.assertTrue(set(train.index).isdisjoint(test.index))
        for s, g in test.groupby('STRING_CODE'):
            self.assertGreater(g['TEST_DATE'].min(), train[train['STRING_CODE'] == s]['TEST_DATE'].max())

    def test_random_split_is_reproducible(self):
        df = prepared()
        a = training01.split_data(df, method='random')
        b = training01.split_data(df, method='random')
        pd.testing.assert_index_equal(a[1].index, b[1].index)
        self.assertEqual(len(a[1]), 24)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            training01.split_data(prepared(), method='foo')


class TestCombinedRegression(unittest.TestCase):

    class Linear:
        """Predicts 2*WH_TEMP + 1, so a deg-2 polynomial fit must reproduce it exactly."""
        def predict(self, X):
            return 2 * X['WH_TEMP'].to_numpy() + 1

    def test_reproduces_smooth_model_and_extrapolates(self):
        X = pd.DataFrame({'WH_TEMP': [80.0, 100.0, 150.0], 'WH_PRESS': [1.0, 2.0, 3.0]})
        grid = np.linspace(70, 110, 9)
        pred = training01.combined_predict(self.Linear(), X, grid, degree=2)
        np.testing.assert_allclose(pred, [161, 201, 301])  # 150 is outside the grid

    def test_missing_sweep_value_falls_back_to_model(self):
        model = mock.Mock()
        model.predict.side_effect = lambda X: np.full(len(X), 5.0)
        X = pd.DataFrame({'WH_TEMP': [80.0, np.nan]})
        pred = training01.combined_predict(model, X, np.linspace(70, 110, 5))
        np.testing.assert_allclose(pred, [5.0, 5.0])

    def test_sweep_grid_uses_training_range(self):
        grid = training01.sweep_grid(np.arange(101.0), n_points=11)
        self.assertEqual(len(grid), 11)
        self.assertAlmostEqual(grid[0], 1.0)
        self.assertAlmostEqual(grid[-1], 99.0)


class TestTrainModel(unittest.TestCase):

    def test_build_model_uses_rf_params(self):
        rf = training01.build_model()[-1]
        self.assertIsInstance(rf, RandomForestRegressor)
        for k, v in training01.RF_PARAMS.items():
            self.assertEqual(rf.get_params()[k], v)
        self.assertEqual(training01.build_model(n_estimators=10)[-1].n_estimators, 10)

    def test_end_to_end_saves_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = os.path.join(tmp, 'welltest.csv')
            make_raw_data().to_csv(csv, index=False)
            model_path = os.path.join(tmp, 'model', 'rf_model.pkl')
            with redirect_stdout(io.StringIO()) as out:
                bundle = training01.train_model(csv, model_path,
                                                model=training01.build_model(n_estimators=20))
            self.assertTrue(os.path.exists(model_path))
            saved = joblib.load(model_path)
        self.assertEqual(saved['features'], FEATURES)
        self.assertEqual(bundle['model_name'], 'Random Forest')
        self.assertIn('Random Forest + polynomial', saved['test_metrics'])
        self.assertIn('Test set', out.getvalue())

        df = training01.data_cleaning(prepared())
        gross = training01.predict_gross_fluid(saved, df)
        self.assertEqual(gross.shape, (len(df),))
        self.assertTrue(np.all(np.isfinite(gross)))


class TestModelSelection(unittest.TestCase):
    """test01.py: algorithm comparison built on training01's data preparation."""

    def test_time_block_folds_follow_time_within_each_string(self):
        train, _ = training01.split_data(prepared())
        train = train.reset_index(drop=True)
        folds = test01.time_block_folds(train, cv=4)
        self.assertEqual(len(folds), 4)
        val = np.concatenate([v for _, v in folds])
        self.assertEqual(sorted(val), list(range(len(train))))   # every row validated once
        for tr, va in folds:
            self.assertTrue(set(tr).isdisjoint(va))
        # first fold = earliest tests of every string
        first = train.iloc[folds[0][1]]
        for s, g in first.groupby('STRING_CODE'):
            later = train[(train['STRING_CODE'] == s) & ~train.index.isin(folds[0][1])]
            self.assertLess(g['TEST_DATE'].max(), later['TEST_DATE'].min())

    def test_main_compares_models_without_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = os.path.join(tmp, 'welltest.csv')
            make_raw_data().to_csv(csv, index=False)
            with mock.patch.object(training01.joblib, 'dump') as dump, redirect_stdout(io.StringIO()) as out:
                results, fitted, params = test01.main(csv, make_plots=False, models=fast_models())
        dump.assert_not_called()
        self.assertEqual(set(fitted), {'Linear', 'Random Forest'})
        self.assertTrue(any(i.endswith('+ polynomial') for i in results.index))
        self.assertIn('Leave-wells-out', out.getvalue())

    def test_feature_importance_is_normalised(self):
        df = training01.data_cleaning(prepared())
        model = training01.build_model(n_estimators=20).fit(df[FEATURES], df[TARGET])
        imp = test01.feature_importance(model, df[FEATURES], df[TARGET])
        self.assertAlmostEqual(imp.sum(), 1.0)
        self.assertEqual(set(imp.index), set(FEATURES))


if __name__ == '__main__':
    unittest.main()
