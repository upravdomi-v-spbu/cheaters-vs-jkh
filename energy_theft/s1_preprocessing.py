"""Stage 1: load, clean, and preprocess meter readings."""

from pathlib import Path
import numpy as np
import pandas as pd


def load_and_preprocess(
    input_path="example.csv",
    *,
    timestamp_column=True,
    interpolation_method="cubic",
    interpolation_limit=3,
    drop_most_missing=4,
    target_noise_scale=5.0,
    random_state=42,
):
    """Load meter data, clean readings, and construct the central target.

    By default the first CSV column is treated as a timestamp/index-like field,
    matching the original notebook. The target is the row-wise sum of cleaned
    meter readings plus Gaussian noise, as in the original experiment.

    Returns (df_clean, X_clean, y_target). For real evaluation, consider loading
    an independently measured central-meter target rather than synthesizing it.
    """
    df = pd.read_csv(input_path)
    readings = df.iloc[:, 1:] if timestamp_column else df.copy()
    readings = readings.select_dtypes(include=[np.number]).copy()
    if readings.empty:
        raise ValueError("No numeric meter-reading columns found in the input CSV.")
    cleaned = readings.interpolate(
        method=interpolation_method, limit=interpolation_limit
    ).clip(lower=0)
    # Retain the notebook's behavior: drop the columns with the most missing data.
    if drop_most_missing and drop_most_missing > 0:
        missing_counts = cleaned.isna().sum().sort_values(ascending=False)
        # Never drop every meter column, even when the input is very narrow.
        n_to_drop = min(drop_most_missing, max(0, len(missing_counts) - 1))
        drop_cols = missing_counts.head(n_to_drop).index
        cleaned = cleaned.drop(columns=drop_cols)
    cleaned = cleaned.fillna(cleaned.mean(numeric_only=True)).fillna(0.0)
    rng = np.random.default_rng(random_state)
    cleaned["target"] = cleaned.sum(axis=1) + rng.normal(
        loc=0.0, scale=target_noise_scale, size=len(cleaned)
    )
    X_clean = cleaned.drop(columns="target")
    y_target = cleaned["target"].to_numpy(dtype=float)
    return cleaned, X_clean, y_target
