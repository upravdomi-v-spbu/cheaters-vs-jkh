"""Stage 3: detection of persistently out-of-order meters."""

import numpy as np
import pandas as pd

def detect_unworking_meters(X_dirty, *, near_zero_fraction=0.98, abs_tol=1e-8):
    """Detect columns with at least `near_zero_fraction` near-zero readings.

    This intentionally targets persistent failures, not temporary blocks or
    periodic zeros. `abs_tol` should be chosen in the units of the meter data.
    Returns one row per meter with the detection diagnostics.
    """
    is_df = isinstance(X_dirty, pd.DataFrame)
    X = np.asarray(X_dirty, dtype=float)
    names = list(X_dirty.columns) if is_df else [f"user_{j}" for j in range(X.shape[1])]
    if X.ndim != 2:
        raise ValueError("X_dirty must be a 2D array or DataFrame.")
    if not 0 <= near_zero_fraction <= 1:
        raise ValueError("near_zero_fraction must be between 0 and 1.")
    near_zero = np.abs(X) <= abs_tol
    frac = near_zero.mean(axis=0) if X.shape[0] else np.ones(X.shape[1])
    max_abs = np.max(np.abs(X), axis=0) if X.shape[0] else np.zeros(X.shape[1])
    detected = frac >= near_zero_fraction
    return pd.DataFrame({
        "user_id": names,
        "near_zero_fraction": frac,
        "max_abs_reading": max_abs,
        "detected_unworking": detected,
    })
