"""Theft-detection estimators and technical evaluation metrics."""

import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd

def fit_meter_alphas(
    X_dirty,
    y,
    *,
    valid_rows=None,
    noise_scale=5.0,
    prior_correction_scale=0.5,
    draws=2000,
    tune=2000,
    chains=4,
    target_accept=0.95,
    random_seed=42,
):
    # Optional heavyweight dependency: imported only when the Bayesian method is used.
    import pymc as pm
    import arviz as az

    if isinstance(X_dirty, pd.DataFrame):
        feature_names = list(X_dirty.columns)
        X_arr = X_dirty.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X_dirty, dtype=float)
        feature_names = [f"user_{j}" for j in range(X_arr.shape[1])]

    y_arr = np.asarray(y, dtype=float).reshape(-1)

    if X_arr.ndim != 2:
        raise ValueError("X_dirty must be a 2D array or DataFrame.")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("X_dirty and y must have the same number of rows.")
    if not np.isfinite(X_arr).all() or not np.isfinite(y_arr).all():
        raise ValueError("X_dirty and y must contain only finite values.")
    if (X_arr < 0).any():
        raise ValueError("Meter readings must be non-negative.")
    if noise_scale <= 0 or prior_correction_scale <= 0:
        raise ValueError("noise_scale and prior_correction_scale must be positive.")

    n_times, n_users = X_arr.shape
    if valid_rows is None:
        mask = np.ones(n_times, dtype=bool)
    else:
        mask = np.asarray(valid_rows, dtype=bool)
        if mask.shape != (n_times,):
            raise ValueError("valid_rows must be a boolean vector with one value per row.")

    X_fit = X_arr[mask]
    y_fit = y_arr[mask]
    if X_fit.shape[0] < max(10, n_users // 2):
        raise ValueError(
            f"Only {X_fit.shape[0]} rows remain for {n_users} users. "
            "Use more valid rows or reduce the model complexity."
        )

    # Columns that are zero throughout the fitting data cannot identify beta.
    all_zero_in_fit = np.all(X_fit == 0, axis=0)
    if all_zero_in_fit.all():
        raise ValueError("All user columns are zero in the selected fitting rows.")

    with pm.Model() as model:
        # beta=1 means no underestimation; beta>1 compensates for low readings.
        correction = pm.HalfNormal(
            "correction",
            sigma=prior_correction_scale,
            shape=n_users,
        )
        beta = pm.Deterministic("beta", 1.0 + correction)

        sigma = pm.HalfNormal("sigma", sigma=noise_scale)
        mu = pm.math.sum(X_fit * beta[None, :], axis=1)

        pm.Normal("central_reading", mu=mu, sigma=sigma, observed=y_fit)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    beta_samples = (
        idata.posterior["beta"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "beta_dim_0")
        .values
    )
    alpha_samples = 1.0 / beta_samples

    rows = []
    for j, name in enumerate(feature_names):
        if all_zero_in_fit[j]:
            rows.append({
                "user_id": name,
                "alpha_median": np.nan,
                "alpha_lower_95": np.nan,
                "alpha_upper_95": np.nan,
                "p_underestimated_5pct": np.nan,
                "p_underestimated_20pct": np.nan,
                "status": "unidentifiable: all zero in fitting rows",
            })
            continue

        samples = alpha_samples[:, j]
        lo, median, hi = np.quantile(samples, [0.025, 0.5, 0.975])
        rows.append({
            "user_id": name,
            "alpha_median": median,
            "alpha_lower_95": lo,
            "alpha_upper_95": hi,
            "p_underestimated_5pct": np.mean(samples < 0.95),
            "p_underestimated_20pct": np.mean(samples < 0.80),
            "status": "estimate; check interval width and diagnostics",
        })

    result = pd.DataFrame(rows)
    diagnostics = az.summary(idata, var_names=["beta", "sigma"], hdi_prob=0.95)
    return result, idata, diagnostics

import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


def fit_scipy_bounded(X, y, *, beta_min=1.0, beta_max=15.0,
                      fit_intercept=False, intercept_bounds=(-10.0, 10.0),
                      maxiter=2000):
    """Least-squares fit with box constraints on beta and optional intercept."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n_features = X.shape[1]

    if fit_intercept:
        initial = np.r_[np.clip(np.ones(n_features), beta_min, beta_max), 0.0]
        bounds = [(beta_min, beta_max)] * n_features + [intercept_bounds]
        def predict(params):
            return X @ params[:-1] + params[-1]
    else:
        initial = np.clip(np.ones(n_features), beta_min, beta_max)
        bounds = [(beta_min, beta_max)] * n_features
        def predict(params):
            return X @ params

    def objective(params):
        residual = y - predict(params)
        return np.mean(residual ** 2)

    result = minimize(
        objective, initial, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter}
    )
    params = result.x
    beta = params[:-1] if fit_intercept else params
    intercept = float(params[-1]) if fit_intercept else 0.0
    return {
        "beta": np.asarray(beta),
        "intercept": intercept,
        "y_pred": predict(params),
        "success": bool(result.success),
        "message": str(result.message),
        "objective": float(result.fun),
    }


def fit_sklearn_linear(X, y, *, fit_intercept=False, positive=False):
    model = LinearRegression(fit_intercept=fit_intercept, positive=positive)
    model.fit(X, y)
    beta = np.asarray(model.coef_, dtype=float).reshape(-1)
    intercept = float(model.intercept_) if fit_intercept else 0.0
    return {
        "beta": beta, "intercept": intercept,
        "y_pred": model.predict(X), "success": True, "message": "ok"
    }


def fit_sklearn_ridge(X, y, *, ridge_penalty=1.0,
                      fit_intercept=False, positive=False):
    model = Ridge(alpha=ridge_penalty, fit_intercept=fit_intercept,
                  positive=positive)
    model.fit(X, y)
    beta = np.asarray(model.coef_, dtype=float).reshape(-1)
    intercept = float(model.intercept_) if fit_intercept else 0.0
    return {
        "beta": beta, "intercept": intercept,
        "y_pred": model.predict(X), "success": True, "message": "ok"
    }



def fit_pymc_bayesian(X, y, *, draws=500, tune=500, chains=2,
                      random_seed=42, noise_scale=5.0,
                      prior_correction_scale=0.5):
    """
    Bayesian estimator using the fit_meter_alphas function defined earlier.
    Returns posterior mean beta and posterior alpha interval as extra fields.
    """
    import numpy as np

    alpha_table, idata, diagnostics = fit_meter_alphas(
        X, y,
        noise_scale=noise_scale,
        prior_correction_scale=prior_correction_scale,
        draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )
    beta_samples = (
        idata.posterior["beta"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "beta_dim_0")
        .values
    )
    beta_mean = beta_samples.mean(axis=0)
    return {
        "beta": beta_mean,
        "intercept": 0.0,
        "y_pred": np.asarray(X, dtype=float) @ beta_mean,
        "success": True,
        "message": "PyMC posterior mean beta",
        "alpha_posterior_lower_95": alpha_table["alpha_lower_95"].to_numpy(),
        "alpha_posterior_upper_95": alpha_table["alpha_upper_95"].to_numpy(),
        "alpha_posterior_median": alpha_table["alpha_median"].to_numpy(),
        "posterior_diagnostics": diagnostics,
    }


def default_comparison_methods(beta_min=1.0, beta_max=15.0, include_bayesian=False):
    """Consistent method registry. Each callable returns beta/intercept/prediction."""
    methods = {
        "OLS_no_intercept": lambda X, y: fit_sklearn_linear(
            X, y, fit_intercept=False, positive=False),
        "OLS_positive_no_intercept": lambda X, y: fit_sklearn_linear(
            X, y, fit_intercept=False, positive=True),
        "OLS_positive_intercept": lambda X, y: fit_sklearn_linear(
            X, y, fit_intercept=True, positive=True),
        "Ridge_0.1_no_intercept": lambda X, y: fit_sklearn_ridge(
            X, y, ridge_penalty=0.1, fit_intercept=False, positive=False),
        "Ridge_1_no_intercept": lambda X, y: fit_sklearn_ridge(
            X, y, ridge_penalty=1.0, fit_intercept=False, positive=False),
        "Ridge_10_no_intercept": lambda X, y: fit_sklearn_ridge(
            X, y, ridge_penalty=10.0, fit_intercept=False, positive=False),
        "Ridge_positive_1_no_intercept": lambda X, y: fit_sklearn_ridge(
            X, y, ridge_penalty=1.0, fit_intercept=False, positive=True),
        "L-BFGS-B_beta_bounds_no_intercept": lambda X, y: fit_scipy_bounded(
            X, y, beta_min=beta_min, beta_max=beta_max,
            fit_intercept=False),
    }
    if include_bayesian:
        methods["PyMC_Bayesian"] = lambda X, y: fit_pymc_bayesian(
            X, y, draws=500, tune=500, chains=2, random_seed=42)
    return methods


def _moving_block_indices(n, block_length, rng):
    """Sample n indices by concatenating randomly selected contiguous blocks."""
    block_length = max(1, min(int(block_length), n))
    starts = rng.integers(0, max(1, n - block_length + 1),
                          size=int(np.ceil(n / block_length)))
    idx = np.concatenate([
        np.arange(s, min(s + block_length, n)) for s in starts
    ])
    return idx[:n]


def compare_alpha_estimators(
    X_dirty, y, alpha_true, *,
    methods=None,
    bootstrap_reps=100,
    block_length=24,
    random_state=42,
    evaluation_mask=None,
):
    """
    Fit estimators and assess implied alpha against known synthetic truth.

    `evaluation_mask` selects columns whose alpha values are identifiable and
    should contribute to alpha-recovery metrics (e.g. exclude truly failed
    meters with alpha_true == 0). All fitted columns are still used by the model.

    Returns
    -------
    method_metrics : one row per method
    user_results : one row per method and user
    Notes:
      * `alpha_true` should be the persistent scaling factor used by the
        corruption generator, not calculated as clean/dirty for zeroed rows.
      * alpha_hat=1/beta_hat is reported only for finite beta_hat > 0.
      * Bootstrap CI is percentile interval from moving-block bootstrap.
      * Predictive metrics here are in-sample, so use a separate time holdout
        if out-of-sample prediction is an evaluation goal.
    """
    is_frame = isinstance(X_dirty, pd.DataFrame)
    names = list(X_dirty.columns) if is_frame else [
        f"user_{j}" for j in range(np.asarray(X_dirty).shape[1])]
    X = np.asarray(X_dirty, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    truth = np.asarray(alpha_true, dtype=float).reshape(-1)

    if X.ndim != 2 or len(y) != X.shape[0]:
        raise ValueError("X_dirty must be 2D and have the same row count as y.")
    if len(truth) != X.shape[1]:
        raise ValueError("alpha_true must have one value per user/column.")
    if evaluation_mask is None:
        eval_mask = np.ones(X.shape[1], dtype=bool)
    else:
        eval_mask = np.asarray(evaluation_mask, dtype=bool).reshape(-1)
        if len(eval_mask) != X.shape[1]:
            raise ValueError("evaluation_mask must have one value per user/column.")
    if methods is None:
        methods = default_comparison_methods()

    rng = np.random.default_rng(random_state)
    metrics_rows, user_rows = [], []

    for method_name, fit_func in methods.items():
        start = time.perf_counter()
        try:
            fit = fit_func(X, y)
            beta = np.asarray(fit["beta"], dtype=float).reshape(-1)
            intercept = float(fit["intercept"])
            pred = np.asarray(fit.get("y_pred", X @ beta + intercept))
            alpha_hat = np.divide(1.0, beta, out=np.full_like(beta, np.nan),
                                   where=np.isfinite(beta) & (beta > 0))

            boot_alpha = []
            boot_beta = []
            boot_failures = 0
            # Bayesian fits already provide posterior intervals; do not run an
            # expensive outer bootstrap around MCMC by default.
            if "alpha_posterior_lower_95" not in fit:
                for _ in range(int(bootstrap_reps)):
                    idx = _moving_block_indices(len(y), block_length, rng)
                    try:
                        bfit = fit_func(X[idx], y[idx])
                        b = np.asarray(bfit["beta"], dtype=float).reshape(-1)
                        a = np.divide(1.0, b, out=np.full_like(b, np.nan),
                                      where=np.isfinite(b) & (b > 0))
                        boot_beta.append(b)
                        boot_alpha.append(a)
                    except Exception:
                        boot_failures += 1

            if boot_alpha:
                boot_alpha = np.asarray(boot_alpha)
                alpha_lo = np.nanquantile(boot_alpha, 0.025, axis=0)
                alpha_hi = np.nanquantile(boot_alpha, 0.975, axis=0)
                beta_boot = np.asarray(boot_beta)
                beta_lo = np.nanquantile(beta_boot, 0.025, axis=0)
                beta_hi = np.nanquantile(beta_boot, 0.975, axis=0)
            else:
                alpha_lo = alpha_hi = beta_lo = beta_hi = np.full(X.shape[1], np.nan)

            # Bayesian intervals are posterior credible intervals, not bootstrap intervals.
            if "alpha_posterior_lower_95" in fit:
                alpha_lo = np.asarray(fit["alpha_posterior_lower_95"], dtype=float)
                alpha_hi = np.asarray(fit["alpha_posterior_upper_95"], dtype=float)
                alpha_hat = np.asarray(fit["alpha_posterior_median"], dtype=float)

            for j, user in enumerate(names):
                user_rows.append({
                    "method": method_name,
                    "user_id": user,
                    "beta_hat": beta[j],
                    "beta_boot_lower_95": beta_lo[j],
                    "beta_boot_upper_95": beta_hi[j],
                    "alpha_true": truth[j],
                    "alpha_hat": alpha_hat[j],
                    "alpha_boot_lower_95": alpha_lo[j],
                    "alpha_boot_upper_95": alpha_hi[j],
                    "alpha_evaluated": bool(eval_mask[j]),
                    "alpha_error": alpha_hat[j] - truth[j]
                        if eval_mask[j] else np.nan,
                    "alpha_abs_error": abs(alpha_hat[j] - truth[j])
                        if eval_mask[j] and np.isfinite(alpha_hat[j]) else np.nan,
                    "alpha_interval_covers_truth": bool(
                        eval_mask[j] and np.isfinite(alpha_lo[j])
                        and np.isfinite(alpha_hi[j])
                        and alpha_lo[j] <= truth[j] <= alpha_hi[j]),
                    "intercept_hat": intercept,
                    "fit_success": fit.get("success", True),
                })

            residual = y - pred
            metrics_rows.append({
                "method": method_name,
                "mse_in_sample": mean_squared_error(y, pred),
                "rmse_in_sample": mean_squared_error(y, pred) ** 0.5,
                "r2_in_sample": r2_score(y, pred),
                "alpha_mae": np.nanmean(np.abs(alpha_hat[eval_mask] - truth[eval_mask])),
                "alpha_rmse": np.sqrt(np.nanmean((alpha_hat[eval_mask] - truth[eval_mask]) ** 2)),
                "alpha_bias": np.nanmean(alpha_hat[eval_mask] - truth[eval_mask]),
                "alpha_interval_coverage": np.nanmean(
                    (alpha_lo[eval_mask] <= truth[eval_mask])
                    & (truth[eval_mask] <= alpha_hi[eval_mask])),
                "mean_alpha_interval_width": np.nanmean(
                    alpha_hi[eval_mask] - alpha_lo[eval_mask]),
                "bootstrap_reps_requested": int(bootstrap_reps),
                "bootstrap_failures": boot_failures,
                "runtime_seconds": time.perf_counter() - start,
                "fit_success": fit.get("success", True),
                "fit_message": fit.get("message", "ok"),
            })
        except Exception as exc:
            metrics_rows.append({
                "method": method_name,
                "mse_in_sample": np.nan, "rmse_in_sample": np.nan,
                "r2_in_sample": np.nan, "alpha_mae": np.nan,
                "alpha_rmse": np.nan, "alpha_bias": np.nan,
                "alpha_interval_coverage": np.nan,
                "mean_alpha_interval_width": np.nan,
                "bootstrap_reps_requested": int(bootstrap_reps),
                "bootstrap_failures": int(bootstrap_reps),
                "runtime_seconds": time.perf_counter() - start,
                "fit_success": False, "fit_message": repr(exc),
            })

    return pd.DataFrame(metrics_rows), pd.DataFrame(user_rows)
