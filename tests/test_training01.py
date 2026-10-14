import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import training01  # noqa: E402

FEATURES = ['WH_CSG_PRESS', 'WH_PRESS', 'WH_TEMP', 'GRAVITY', 'SURF_CHOKE', 'SEP_PRESS']
TARGET = 'GROSS_FLUID'


def make_raw_data(n=50, seed=0):
    """Synthetic well test data including extra columns that load_data should drop."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'TEST_DATE': pd.date_range('2024-01-01', periods=n).astype(str),
        'WH_CSG_PRESS': rng.uniform(100, 1500, n),
        'WH_PRESS': rng.uniform(100, 3000, n),
        'WH_TEMP': rng.uniform(60, 200, n),
        'GRAVITY': rng.uniform(20, 40, n),
        'GROSS_FLUID': rng.uniform(100, 5000, n),
        'SURF_CHOKE': rng.uniform(8, 128, n),
        'SEP_PRESS': rng.uniform(50, 300, n),
        'OIL_RATE': rng.uniform(10, 1000, n),
    })


class TestLoadData(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.raw = make_raw_data(n=50)
        self.csv_path = os.path.join(self.tmpdir.name, 'welltest.csv')
        self.raw.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_csv_returns_80_20_split(self):
        X_train, X_test, y_train, y_test = training01.load_data(self.csv_path)
        self.assertEqual(len(X_train), 40)
        self.assertEqual(len(X_test), 10)
        self.assertEqual(len(y_train), 40)
        self.assertEqual(len(y_test), 10)

    def test_csv_selects_feature_and_target_columns(self):
        X_train, X_test, y_train, y_test = training01.load_data(self.csv_path)
        self.assertEqual(list(X_train.columns), FEATURES)
        self.assertEqual(list(X_test.columns), FEATURES)
        self.assertEqual(y_train.name, TARGET)
        self.assertNotIn('OIL_RATE', X_train.columns)
        self.assertNotIn('TEST_DATE', X_train.columns)

    def test_split_is_reproducible(self):
        first = training01.load_data(self.csv_path)
        second = training01.load_data(self.csv_path)
        for a, b in zip(first, second):
            pd.testing.assert_index_equal(a.index, b.index)

    def test_train_and_test_do_not_overlap(self):
        X_train, X_test, _, _ = training01.load_data(self.csv_path)
        self.assertTrue(set(X_train.index).isdisjoint(X_test.index))

    def test_xlsx_uses_read_excel(self):
        with mock.patch.object(training01.pd, 'read_excel', return_value=self.raw) as read_excel:
            X_train, X_test, _, _ = training01.load_data('welltest.xlsx')
        read_excel.assert_called_once_with('welltest.xlsx')
        self.assertEqual(len(X_train) + len(X_test), 50)

    def test_missing_required_column_raises(self):
        path = os.path.join(self.tmpdir.name, 'bad.csv')
        self.raw.drop(columns='SEP_PRESS').to_csv(path, index=False)
        with self.assertRaises(KeyError):
            training01.load_data(path)

    def test_unsupported_extension_raises(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported file type'):
            training01.load_data('welltest.txt')


class TestDataCleaning(unittest.TestCase):

    def make_df(self, rows):
        return pd.DataFrame(rows, columns=FEATURES + [TARGET])

    def test_filters_out_of_range_rows(self):
        df = self.make_df([
            # WH_CSG_PRESS, WH_PRESS, WH_TEMP, GRAVITY, SURF_CHOKE, SEP_PRESS, GROSS_FLUID
            [500, 1000, 100, 30, 64, 100, 1000],   # keep
            [500, 3501, 100, 30, 64, 100, 1000],   # WH_PRESS too high
            [500, 1000, 100, 30, 64, 100, 99],     # GROSS_FLUID too low
            [500, 1000, 100, 30, 129, 100, 1000],  # SURF_CHOKE too high
        ])
        cleaned = training01.data_cleaning(df)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]['WH_PRESS'], 1000)

    def test_boundary_values_are_kept(self):
        df = self.make_df([[500, 3500, 100, 30, 128, 100, 100]])
        cleaned = training01.data_cleaning(df)
        self.assertEqual(len(cleaned), 1)

    def test_imputes_missing_values_with_median(self):
        df = self.make_df([
            [100, 1000, 100, 30, 64, 100, 1000],
            [200, 1000, 120, 30, 64, 100, 1000],
            [np.nan, 1000, np.nan, 30, 64, 100, 1000],
            [600, 1000, 200, 30, 64, 100, 1000],
        ])
        cleaned = training01.data_cleaning(df)
        self.assertFalse(cleaned.isna().any().any())
        self.assertEqual(cleaned.loc[2, 'WH_CSG_PRESS'], 200)  # median of 100, 200, 600
        self.assertEqual(cleaned.loc[2, 'WH_TEMP'], 120)       # median of 100, 120, 200

    def test_rows_with_nan_in_filter_columns_are_dropped(self):
        # NaN comparisons are False, so these rows are removed before imputation.
        df = self.make_df([
            [500, 1000, 100, 30, 64, 100, 1000],
            [500, np.nan, 100, 30, 64, 100, 1000],
            [500, 1000, 100, 30, np.nan, 100, 1000],
            [500, 1000, 100, 30, 64, 100, np.nan],
        ])
        cleaned = training01.data_cleaning(df)
        self.assertEqual(len(cleaned), 1)

    def test_preserves_columns_and_resets_index(self):
        df = self.make_df([
            [500, 1000, 100, 30, 64, 100, 1000],
            [500, 9999, 100, 30, 64, 100, 1000],
            [500, 1000, 100, 30, 64, 100, 2000],
        ])
        cleaned = training01.data_cleaning(df)
        self.assertEqual(list(cleaned.columns), FEATURES + [TARGET])
        self.assertEqual(list(cleaned.index), [0, 1])

    def test_does_not_modify_input(self):
        df = self.make_df([
            [np.nan, 1000, 100, 30, 64, 100, 1000],
            [400, 1000, 100, 30, 64, 100, 1000],
            [500, 9999, 100, 30, 64, 100, 1000],
        ])
        original = df.copy()
        training01.data_cleaning(df)
        pd.testing.assert_frame_equal(df, original)

    def test_all_nan_column_is_kept(self):
        df = self.make_df([
            [np.nan, 1000, 100, 30, 64, 100, 1000],
            [np.nan, 1000, 100, 30, 64, 100, 1000],
        ])
        cleaned = training01.data_cleaning(df)
        self.assertEqual(list(cleaned.columns), FEATURES + [TARGET])
        # No median exists for an all-NaN column, so it stays NaN rather than being filled with 0.
        self.assertTrue(cleaned['WH_CSG_PRESS'].isna().all())


class TestTrainModel(unittest.TestCase):

    def setUp(self):
        raw = make_raw_data(n=60, seed=1)
        self.X = raw[FEATURES]
        self.y = raw[TARGET]
        # Never write to the real model directory during tests.
        patcher = mock.patch.object(training01.joblib, 'dump')
        self.dump = patcher.start()
        self.addCleanup(patcher.stop)

    def train(self):
        with redirect_stdout(io.StringIO()) as out:
            model = training01.train_model(self.X[:48], self.y[:48], self.X[48:], self.y[48:])
        return model, out.getvalue()

    def test_returns_fitted_random_forest(self):
        model, _ = self.train()
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.n_features_in_, len(FEATURES))
        self.assertEqual(model.predict(self.X[48:]).shape, (12,))

    def test_prints_metrics(self):
        _, output = self.train()
        self.assertIn('Mean Squared Error:', output)
        self.assertIn('R^2 Score:', output)

    def test_saves_model(self):
        model, _ = self.train()
        self.dump.assert_called_once()
        saved_model, saved_path = self.dump.call_args.args
        self.assertIs(saved_model, model)
        self.assertTrue(saved_path.endswith(os.path.join('model', 'rf_model.pkl')))

    def test_is_deterministic(self):
        first, _ = self.train()
        second, _ = self.train()
        np.testing.assert_array_equal(first.predict(self.X), second.predict(self.X))


class TestMain(unittest.TestCase):

    def test_cleans_train_and_drops_nan_from_test(self):
        X_train = pd.DataFrame([
            [500, 1000, 100, 30, 64, 100],
            [500, 9999, 100, 30, 64, 100],      # filtered out by WH_PRESS
            [np.nan, 1000, 100, 30, 64, 100],   # imputed
        ], columns=FEATURES)
        y_train = pd.Series([1000, 1000, 2000], name=TARGET)
        X_test = pd.DataFrame([
            [500, 1000, 100, 30, 64, 100],
            [np.nan, 1000, 100, 30, 64, 100],   # dropped
        ], columns=FEATURES, index=[10, 11])
        y_test = pd.Series([1500, 1500], name=TARGET, index=[10, 11])

        with mock.patch.object(training01, 'load_data',
                               return_value=(X_train, X_test, y_train, y_test)), \
             mock.patch.object(training01, 'train_model') as train_model:
            training01.main()

        Xtr, ytr, Xte, yte = train_model.call_args.args
        self.assertEqual(len(Xtr), 2)
        self.assertFalse(Xtr.isna().any().any())
        self.assertEqual(list(Xtr.columns), FEATURES)
        self.assertEqual(ytr.name, TARGET)
        self.assertEqual(len(Xte), 1)
        self.assertEqual(list(yte), [1500])


if __name__ == '__main__':
    unittest.main()
