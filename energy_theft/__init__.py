"""Modular energy theft detection experiment."""

from .s1_preprocessing import load_and_preprocess
from .s2_synthetic_data import make_corrupted_meter_data
from .s3_out_of_order_detection import detect_unworking_meters
from .s4_estimators import default_comparison_methods, compare_alpha_estimators
from .s5_experiment_runner import run_auto_test, run_repeated_auto_tests, save_comparison_results

__all__ = [
    "load_and_preprocess", "make_corrupted_meter_data", "detect_unworking_meters",
    "default_comparison_methods", "compare_alpha_estimators", "run_auto_test",
    "run_repeated_auto_tests", "save_comparison_results",
]
