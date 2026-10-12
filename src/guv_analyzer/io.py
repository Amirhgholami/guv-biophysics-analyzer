import numpy as np
import pandas as pd


def load_fluorescence_csv(path):
    """Load and validate fluorescence data from a CSV or Excel file."""
    path = str(path)
    if path.lower().endswith(".csv"):
        data = pd.read_csv(path)
    elif path.lower().endswith((".xlsx", ".xls")):
        data = pd.read_excel(path)
    else:
        raise ValueError(
            "Unsupported file format. Use CSV or Excel (.xlsx, .xls)."
        )

    required = {"time_s", "fluorescence"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    data = data[["time_s", "fluorescence"]].copy()

    for column in ["time_s", "fluorescence"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.isna().any().any():
        raise ValueError("Data contains missing or non-numeric values.")

    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Data contains infinite values.")

    if len(data) < 5:
        raise ValueError("At least 5 data points are required.")

    time_s = data["time_s"].to_numpy()
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("Time values must be strictly increasing.")

    return data
