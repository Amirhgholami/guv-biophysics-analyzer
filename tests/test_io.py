import pandas as pd
import pytest

from src.guv_analyzer.io import load_fluorescence_csv


def test_load_valid_csv(tmp_path):
    path = tmp_path / "valid.csv"
    pd.DataFrame({
        "time_s": [0, 1, 2, 3, 4],
        "fluorescence": [100, 102, 105, 108, 110],
    }).to_csv(path, index=False)

    data = load_fluorescence_csv(path)

    assert list(data.columns) == ["time_s", "fluorescence"]
    assert len(data) == 5


def test_missing_column_raises_error(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({
        "time_s": [0, 1, 2, 3, 4],
        "signal": [100, 102, 105, 108, 110],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_fluorescence_csv(path)


def test_non_increasing_time_raises_error(tmp_path):
    path = tmp_path / "invalid_time.csv"
    pd.DataFrame({
        "time_s": [0, 1, 2, 2, 4],
        "fluorescence": [100, 102, 105, 108, 110],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="strictly increasing"):
        load_fluorescence_csv(path)
