"""Synthetic corruption generators for thief-like and failed-meter behavior."""

import numpy as np
import pandas as pd

def distort_columns_by_multiplier(X, col_share=0.2, random_state=None):
    """Домножает случайные столбцы матрицы X на случайный коэффициент из [0, 1)."""
    if random_state is not None:
        np.random.seed(random_state)

    X_distorted = X.copy()
    n_samples, n_features = X_distorted.shape

    # Находим индексы столбцов, которые будем искажать
    n_distort = int(n_features * col_share)
    distort_indices = np.random.choice(n_features, size=n_distort, replace=False)

    # Генерируем случайные коэффициенты для каждого выбранного столбца
    multipliers = np.random.uniform(0.0, 1.0, size=n_distort)

    # Применяем умножение
    if isinstance(X_distorted, pd.DataFrame):
        for idx, mult in zip(distort_indices, multipliers):
            X_distorted.iloc[:, idx] = X_distorted.iloc[:, idx] * mult
    else:
        for idx, mult in zip(distort_indices, multipliers):
            X_distorted[:, idx] = X_distorted[:, idx] * mult

    return X_distorted

def zero_consecutive_elements(X, col_share=0.2, drop_share=0.1, random_state=None):
    """В случайных столбцах зануляет заданную долю элементов, идущих подряд."""
    if random_state is not None:
        np.random.seed(random_state)

    X_distorted = X.copy()
    n_samples, n_features = X_distorted.shape

    # Сколько столбцов нужно исказить
    n_distort = int(n_features * col_share)
    distort_indices = np.random.choice(n_features, size=n_distort, replace=False)

    # Сколько элементов подряд нужно занулить в столбце
    block_size = max(1, int(n_samples * drop_share))

    # Преобразуем в numpy для быстрого индексирования, если это DataFrame
    is_df = isinstance(X_distorted, pd.DataFrame)
    arr = X_distorted.values.copy() if is_df else X_distorted.copy()

    for col_idx in distort_indices:
        # Случайно выбираем начало блока по строкам
        max_start = n_samples - block_size
        start_row = (
            np.random.randint(0, max_start + 1) if max_start > 0 else 0
        )

        arr[start_row : start_row + block_size, col_idx] = 0.0

    return pd.DataFrame(arr, columns=X.columns, index=X.index) if is_df else arr

def zero_every_kth_element(X, col_share=0.2, k=None, random_state=None):
    """В случайных столбцах обнуляет каждый k-й элемент.

    Если k=None, шаг выбирается случайно для каждого столбца.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X_distorted = X.copy()
    n_samples, n_features = X_distorted.shape

    # Выбираем столбцы, которые будем искажать
    n_distort = int(n_features * col_share)
    distort_indices = np.random.choice(n_features, size=n_distort, replace=False)

    is_df = isinstance(X_distorted, pd.DataFrame)
    arr = X_distorted.values.copy() if is_df else X_distorted.copy()

    for idx in distort_indices:
        # Если k не задан, генерируем случайный шаг
        current_k = (
            k
            if k is not None
            else np.random.randint(2, max(3, n_samples // 2))
        )

        # Обнуляем элементы через заданный шаг по строкам
        arr[::current_k, idx] = 0.0

    return (
        pd.DataFrame(arr, columns=X.columns, index=X.index)
        if is_df
        else arr
    )

def make_corrupted_meter_data(
    X_clean, *,
    unworking_share=0.05,
    scale_share=0.3,
    scale_range=(0.2, 0.9),
    block_share=0.05,
    block_fraction=0.20,
    periodic_share=0.05,
    every_k=3,
    random_state=42,
):
    """Return corrupted readings, true persistent alphas, and metadata.

    `unworking_share` selects permanently failed meters: their entire observed
    columns are set to zero and alpha_true is 0. The other corruption stages
    are applied only to the remaining working meters. Stage selections may
    overlap with one another.
    """
    is_df = isinstance(X_clean, pd.DataFrame)
    columns = list(X_clean.columns) if is_df else [
        f"user_{j}" for j in range(np.asarray(X_clean).shape[1])]
    X = np.asarray(X_clean, dtype=float).copy()
    n_rows, n_cols = X.shape
    if n_cols == 0:
        raise ValueError("X_clean must contain at least one meter column.")
    rng = np.random.default_rng(random_state)

    def sample_columns(pool, share):
        pool = np.asarray(pool, dtype=int)
        if len(pool) == 0 or share <= 0:
            return np.array([], dtype=int)
        count = min(len(pool), max(1, int(round(n_cols * share))))
        return rng.choice(pool, size=count, replace=False)

    all_cols = np.arange(n_cols)
    unworking_cols = sample_columns(all_cols, unworking_share)
    alpha_true = np.ones(n_cols, dtype=float)
    alpha_true[unworking_cols] = 0.0
    X[:, unworking_cols] = 0.0
    working_cols = np.setdiff1d(all_cols, unworking_cols)

    scaled_cols = sample_columns(working_cols, scale_share)
    multipliers = rng.uniform(scale_range[0], scale_range[1], size=len(scaled_cols))
    alpha_true[scaled_cols] = multipliers
    X *= alpha_true[None, :]

    block_mask = np.zeros_like(X, dtype=bool)
    block_cols = sample_columns(working_cols, block_share)
    block_size = max(1, int(n_rows * block_fraction)) if n_rows else 0
    for j in block_cols:
        if block_size > 0:
            start = int(rng.integers(0, max(1, n_rows - block_size + 1)))
            block_mask[start:start + block_size, j] = True
    X[block_mask] = 0.0

    periodic_mask = np.zeros_like(X, dtype=bool)
    periodic_cols = sample_columns(working_cols, periodic_share)
    if every_k is not None and every_k >= 1:
        for j in periodic_cols:
            periodic_mask[::every_k, j] = True
        X[periodic_mask] = 0.0

    dirty = pd.DataFrame(X, columns=columns, index=X_clean.index) if is_df else X
    metadata = {
        "unworking_columns": [columns[j] for j in unworking_cols],
        "unworking_mask": np.isin(all_cols, unworking_cols),
        "scaled_columns": [columns[j] for j in scaled_cols],
        "scale_multipliers": {columns[j]: alpha_true[j] for j in scaled_cols},
        "block_columns": [columns[j] for j in block_cols],
        "periodic_columns": [columns[j] for j in periodic_cols],
        "block_mask": block_mask,
        "periodic_mask": periodic_mask,
    }
    return dirty, alpha_true, metadata
