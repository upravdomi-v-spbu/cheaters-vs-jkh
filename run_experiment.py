"""Run a small energy-theft detection benchmark from the project root."""

from energy_theft import load_and_preprocess, default_comparison_methods, run_auto_test, save_comparison_results


def main():
    _, X_clean, y_target = load_and_preprocess("example.csv", random_state=42)
    methods = default_comparison_methods(
        beta_min=1.0, beta_max=15.0, include_bayesian=False
    )
    result = run_auto_test(
        X_clean,
        y_target,
        methods=methods,
        trial_id=0,
        corruption_kwargs={
            "unworking_share": 0.05,
            "scale_share": 0.05,
            "scale_range": (0.2, 1.0),
            "block_share": 0.05,
            "block_fraction": 0.20,
            "periodic_share": 0.05,
            "every_k": 3,
        },
        bootstrap_reps=20,
        block_length=24,
        random_state=42,
    )
    print(result["trial_summary"].to_string(index=False))
    print(result["method_metrics"].sort_values(["pipeline_variant", "alpha_mae"]).to_string(index=False))
    save_comparison_results(result, output_dir="model_comparison_results", append=False)


if __name__ == "__main__":
    main()
