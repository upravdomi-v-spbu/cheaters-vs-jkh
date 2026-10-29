# Energy theft detection — modular version

The notebook has been split into stages with explicit responsibilities:

| Stage | File | Responsibility |
|---|---|---|
| S1 | `energy_theft/s1_preprocessing.py` | Load CSV, clean/interpolate readings, construct the experiment target |
| S2 | `energy_theft/s2_synthetic_data.py` | Generate synthetic theft-like scaling, temporary blocks, periodic zeros, and permanently failed meters |
| S3 | `energy_theft/s3_out_of_order_detection.py` | Detect persistently near-zero meter columns from observed readings only |
| S4A+ | `energy_theft/s4_estimators.py` | Estimators (bounded SciPy, sklearn linear/Ridge, Bayesian PyMC), bootstrap uncertainty, and technical metrics |
| S5 | `energy_theft/s5_experiment_runner.py` | Run variants/trials, compare adjusted vs unadjusted pipelines, and save result tables |

## Setup

Use Python 3.10+ (PyMC may have stricter version constraints on some platforms):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Put `example.csv` in the project root, then run:

```bash
python run_experiment.py
```

Bayesian sampling is disabled by default in `run_experiment.py` because it is much slower. Set `include_bayesian=True` in the method registry to include it.

## Use from Python

```python
from energy_theft import (
    load_and_preprocess, default_comparison_methods,
    run_auto_test, save_comparison_results,
)

_, X_clean, y_target = load_and_preprocess("example.csv", random_state=42)
methods = default_comparison_methods(include_bayesian=False)
result = run_auto_test(X_clean, y_target, methods=methods, random_state=42)
save_comparison_results(result, append=False)
```

`run_auto_test` returns `method_metrics`, `user_results`, `detection_results`, `trial_summary`, plus detailed trial data. `run_repeated_auto_tests` supports repeated synthetic trials. The S5 equal-share adjustment is retained as a baseline assumption, not a physically guaranteed correction.

## Notes and caveats

- The original notebook creates `target` from the row sum plus noise. This behavior is preserved for reproducibility; for a real deployment, use an independently measured aggregate/central meter target.
- S2 generator ground-truth labels are used for evaluation only. S3 detection relies on observed readings.
- The estimator code uses `alpha_hat = 1 / beta_hat`; zero/negative beta values are invalid alpha estimates. Keep alpha-recovery metrics separate from prediction metrics.
- `s4_estimators.py` contains all the estimator variants currently in the notebook. Add a new variant there and register it in `default_comparison_methods`; the S5 runner can then evaluate it through the shared interface.
