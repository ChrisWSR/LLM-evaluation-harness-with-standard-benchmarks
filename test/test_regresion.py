"""
Regression guard: fails CI if any model's latency exceeds threshold.
Run after benchmark.py to catch performance regressions in PRs.
"""

import argparse
import sys
import pandas as pd


def check_regressions(results_dir: str, threshold_ms: float):
    df = pd.read_csv(f"{results_dir}/benchmark_results.csv")
    bs1 = df[df["batch_size"] == 1]

    failures = bs1[bs1["mean_latency_ms"] > threshold_ms]
    if not failures.empty:
        print(f"\n❌ REGRESSION DETECTED — latency > {threshold_ms}ms threshold:\n")
        print(failures[["model_name", "backend", "mean_latency_ms"]].to_string(index=False))
        sys.exit(1)
    else:
        print(f"✅ All models within {threshold_ms}ms threshold. No regressions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--threshold-ms", type=float, default=200.0)
    args = parser.parse_args()
    check_regressions(args.results_dir, args.threshold_ms)