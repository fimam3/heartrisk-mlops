import numpy as np
import pandas as pd
import pytest

from heartrisk import config, data
from conftest import make_synthetic_raw, write_raw_file


def test_load_raw_parses_question_marks_as_nan(tmp_path):
    df = make_synthetic_raw(60, seed=3)
    f = tmp_path / "raw.data"
    write_raw_file(df, f)
    loaded = data.load_raw(f)
    assert list(loaded.columns) == config.RAW_COLUMNS
    assert loaded["ca"].isna().sum() == df["ca"].isna().sum()
    assert pd.api.types.is_numeric_dtype(loaded["thal"])


def test_clean_binarises_target_and_drops_num(raw_df):
    out = data.clean(raw_df)
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert (out[config.TARGET] == (raw_df["num"] > 0).astype(int).values[: len(out)]).all()


def test_clean_removes_exact_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
    assert len(data.clean(doubled)) == len(data.clean(raw_df))


def test_clean_keeps_missing_for_pipeline_imputation(raw_df):
    out = data.clean(raw_df)
    assert out[["ca", "thal"]].isna().sum().sum() > 0


def test_clean_coerces_bad_tokens_to_nan(raw_df):
    bad = raw_df.copy().astype(object)
    bad.loc[0, "chol"] = "n/a"
    out = data.clean(bad)
    assert out["chol"].isna().sum() == 1


def test_validate_schema_raises_on_missing_column(raw_df):
    with pytest.raises(ValueError, match="Missing required columns"):
        data.validate_schema(raw_df.drop(columns=["age"]), config.RAW_COLUMNS)


def test_split_is_stratified_and_disjoint(clean_df):
    X_tr, X_te, y_tr, y_te = data.split_xy(clean_df)
    assert len(X_tr) + len(X_te) == len(clean_df)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    assert list(X_tr.columns) == config.FEATURES
    assert X_tr.dtypes.eq(np.float64).all()


def test_save_and_load_roundtrip(clean_df, tmp_path):
    p = data.save_clean(clean_df, tmp_path / "sub" / "clean.csv")
    pd.testing.assert_frame_equal(data.load_clean(p), clean_df, check_dtype=False)
