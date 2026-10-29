"""Stage 5: orchestrate estimator variants and calculate product-level metrics."""

from pathlib import Path
import numpy as np
import pandas as pd
from .s2_synthetic_data import make_corrupted_meter_data
from .s3_out_of_order_detection import detect_unworking_meters
from .s4_estimators import default_comparison_methods, compare_alpha_estimators

def run_auto_test(
    X_clean, y, *,
    methods=None,
    trial_id=0,
    corruption_kwargs=None,
    bootstrap_reps=100,
    block_length=24,
    random_state=42,
    near_zero_fraction=0.98,
    near_zero_abs_tol=1e-8,
):
    """Run unadjusted and equal-share failed-meter baselines on one trial.

    Equal-share baseline:
      1) detect near-zero columns from X_dirty;
      2) let k be the number detected and n be the original number of meters;
      3) use y_adjusted = y * (1 - k/n);
      4) exclude detected columns and fit the same estimators.

    Returns comparison metrics, per-user results, detection results, adjusted
    target, corruption metadata, and a one-row trial summary.
    """
    corruption_kwargs = dict(corruption_kwargs or {})
    dirty, alpha_true, corruption_metadata = make_corrupted_meter_data(
        X_clean, random_state=random_state, **corruption_kwargs
    )
    y = np.asarray(y, dtype=float).reshape(-1)
    X_arr = np.asarray(dirty, dtype=float)
    names = list(dirty.columns) if isinstance(dirty, pd.DataFrame) else [
        f"user_{j}" for j in range(X_arr.shape[1])]
    alpha_true = np.asarray(alpha_true, dtype=float)
    n_rows, n_total = X_arr.shape
    if len(y) != n_rows:
        raise ValueError("y must have the same number of rows as X_clean.")
    if methods is None:
        methods = default_comparison_methods()

    # Detection uses only the corrupted readings, not the known synthetic labels.
    detection_results = detect_unworking_meters(
        dirty, near_zero_fraction=near_zero_fraction, abs_tol=near_zero_abs_tol
    )
    detected_mask = detection_results["detected_unworking"].to_numpy(dtype=bool)
    detected_names = [names[j] for j in np.flatnonzero(detected_mask)]
    working_mask_true = alpha_true > 0
    n_detected = int(detected_mask.sum())
    y_adjusted = y * (1.0 - n_detected / n_total)
    keep_mask = ~detected_mask
    kept_names = [names[j] for j in np.flatnonzero(keep_mask)]

    true_unworking = np.asarray(corruption_metadata["unworking_mask"], dtype=bool)
    tp = int(np.sum(detected_mask & true_unworking))
    fp = int(np.sum(detected_mask & ~true_unworking))
    fn = int(np.sum(~detected_mask & true_unworking))
    detection_precision = tp / (tp + fp) if (tp + fp) else np.nan
    detection_recall = tp / (tp + fn) if (tp + fn) else np.nan
    detection_f1 = (2 * detection_precision * detection_recall /
                    (detection_precision + detection_recall)
                    if np.isfinite(detection_precision) and
                    np.isfinite(detection_recall) and
                    (detection_precision + detection_recall) > 0 else np.nan)
    detection_results["true_unworking"] = true_unworking
    detection_results["detection_correct"] = (
        detection_results["detected_unworking"].to_numpy() == true_unworking)
    detection_results.insert(0, "trial_id", trial_id)
    detection_results["random_state"] = random_state

    # Variant 1: original pipeline, all columns and unadjusted target.
    raw_metrics, raw_users = compare_alpha_estimators(
        dirty, y, alpha_true,
        methods=methods,
        bootstrap_reps=bootstrap_reps,
        block_length=block_length,
        random_state=random_state + 1,
        evaluation_mask=working_mask_true,
    )
    raw_metrics["pipeline_variant"] = "unadjusted_all_meters"
    raw_users["pipeline_variant"] = "unadjusted_all_meters"
    raw_metrics["target_adjustment"] = "none"
    raw_metrics["n_meters_used"] = n_total

    # Variant 2: equal-share correction and detected-column exclusion.
    if not keep_mask.any():
        raise ValueError("All meters were detected as unworking; no predictors remain.")
    X_kept = dirty.loc[:, keep_mask] if isinstance(dirty, pd.DataFrame) else X_arr[:, keep_mask]
    alpha_kept = alpha_true[keep_mask]
    kept_eval_mask = alpha_kept > 0
    adjusted_metrics, adjusted_users = compare_alpha_estimators(
        X_kept, y_adjusted, alpha_kept,
        methods=methods,
        bootstrap_reps=bootstrap_reps,
        block_length=block_length,
        random_state=random_state + 2,
        evaluation_mask=kept_eval_mask,
    )
    adjusted_metrics["pipeline_variant"] = "equal_share_adjusted"
    adjusted_users["pipeline_variant"] = "equal_share_adjusted"
    adjusted_metrics["target_adjustment"] = f"y*(1-{n_detected}/{n_total})"
    adjusted_metrics["n_meters_used"] = int(keep_mask.sum())

    for frame in (raw_metrics, adjusted_metrics):
        frame["trial_id"] = trial_id
        frame["random_state"] = random_state
        frame["detected_unworking_count"] = n_detected
        frame["detection_precision"] = detection_precision
        frame["detection_recall"] = detection_recall
        frame["detection_f1"] = detection_f1
    for frame in (raw_users, adjusted_users):
        frame["trial_id"] = trial_id
        frame["random_state"] = random_state
        frame["detected_unworking_count"] = n_detected

    method_metrics = pd.concat([raw_metrics, adjusted_metrics], ignore_index=True, sort=False)
    user_results = pd.concat([raw_users, adjusted_users], ignore_index=True, sort=False)
    trial_summary = pd.DataFrame([{
        "trial_id": trial_id,
        "random_state": random_state,
        "n_rows": n_rows,
        "n_users": n_total,
        "n_unworking_true": int(true_unworking.sum()),
        "n_unworking_detected": n_detected,
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "detection_f1": detection_f1,
        "n_scaled_users": len(corruption_metadata["scaled_columns"]),
        "n_block_corrupted_users": len(corruption_metadata["block_columns"]),
        "n_periodic_corrupted_users": len(corruption_metadata["periodic_columns"]),
        "equal_share_fraction_removed": n_detected / n_total,
    }])

    return {
        "X_dirty": dirty,
        "alpha_true": pd.Series(alpha_true, index=names, name="alpha_true"),
        "corruption_metadata": corruption_metadata,
        "detection_results": detection_results,
        "y_adjusted": pd.Series(y_adjusted, index=(dirty.index if isinstance(dirty, pd.DataFrame) else None), name="y_adjusted"),
        "detected_unworking_columns": detected_names,
        "kept_columns": kept_names,
        "method_metrics": method_metrics,
        "user_results": user_results,
        "trial_summary": trial_summary,
    }


def run_repeated_auto_tests(
    X_clean, y, *, n_trials=10, methods=None,
    corruption_kwargs=None, bootstrap_reps=100, block_length=24,
    base_seed=42, near_zero_fraction=0.98, near_zero_abs_tol=1e-8,
):
    """Run repeated trials and concatenate all result tables, including detection."""
    metric_tables, user_tables, detection_tables, trial_tables = [], [], [], []
    last_run = None
    for trial in range(n_trials):
        result = run_auto_test(
            X_clean, y,
            methods=methods,
            trial_id=trial,
            corruption_kwargs=corruption_kwargs,
            bootstrap_reps=bootstrap_reps,
            block_length=block_length,
            random_state=base_seed + trial,
            near_zero_fraction=near_zero_fraction,
            near_zero_abs_tol=near_zero_abs_tol,
        )
        metric_tables.append(result["method_metrics"])
        user_tables.append(result["user_results"])
        detection = result["detection_results"].copy()
        detection["trial_id"] = trial
        detection["random_state"] = base_seed + trial
        detection_tables.append(detection)
        trial_tables.append(result["trial_summary"])
        last_run = result

    return {
        "method_metrics": pd.concat(metric_tables, ignore_index=True),
        "user_results": pd.concat(user_tables, ignore_index=True),
        "detection_results": pd.concat(detection_tables, ignore_index=True),
        "trial_summary": pd.concat(trial_tables, ignore_index=True),
        "last_run": last_run,
    }

# Save this run's result tables. Set append=True for accumulating trials.
def save_comparison_results(result, output_dir="model_comparison_results", append=True):
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "method_metrics.csv": result["method_metrics"],
        "user_results.csv": result["user_results"],
        "trial_summary.csv": result["trial_summary"],
    }
    if "detection_results" in result:
        tables["detection_results.csv"] = result["detection_results"]
    for filename, table in tables.items():
        file_path = output_dir / filename
        write_header = not (append and file_path.exists())
        table.to_csv(file_path, mode="a" if append else "w",
                     header=write_header, index=False)
    return str(output_dir.resolve())

