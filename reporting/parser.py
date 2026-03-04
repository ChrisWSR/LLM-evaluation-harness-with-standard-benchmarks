"""
parser.py — Results Parser
===========================
Reads all raw JSON result files from benchmarks/results/ and normalises
them into a clean pandas DataFrame that reporter.py can consume directly.

Each row in the final DataFrame represents one (model, task) pair:
  model | task | score | stderr | metric | timestamp
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# Canonical metric key per task — same map used in runner.py
TASK_METRICS = {
    "mmlu"          : "acc",
    "gsm8k"         : "exact_match",
    "humaneval"     : "pass@1",
    "arc_challenge" : "acc_norm",
    "gpqa"          : "acc",
}

# Human-readable task labels for charts
TASK_LABELS = {
    "mmlu"          : "MMLU",
    "gsm8k"         : "GSM8K",
    "humaneval"     : "HumanEval",
    "arc_challenge" : "ARC-C",
    "gpqa"          : "GPQA",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Read raw JSON files from disk
# ══════════════════════════════════════════════════════════════════════════════

def parse_results(results_dir: Path) -> list[dict]:
    """
    Walk results_dir and load every metadata.json it finds.

    Directory structure expected:
      benchmarks/results/
        phi-3-mini/
          metadata.json
          raw_20240101_120000.json
        mistral-7b/
          metadata.json
          ...

    Returns:
        List of metadata dicts, one per model. Example:
        [
          {
            "model"     : "phi-3-mini",
            "tasks"     : ["mmlu", "gsm8k"],
            "timestamp" : "2024-01-01T12:00:00",
            "results"   : { "mmlu": {"acc": 0.61, ...}, ... }
          },
          ...
        ]
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        log.error(f"Results directory not found: {results_dir}")
        return []

    loaded = []

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        meta_path = model_dir / "metadata.json"
        if not meta_path.exists():
            # Fall back to the latest raw_*.json if no metadata.json
            raw_files = sorted(model_dir.glob("raw_*.json"))
            if not raw_files:
                log.warning(f"No results found in {model_dir} — skipping.")
                continue
            meta_path = raw_files[-1]
            log.debug(f"No metadata.json found, using: {meta_path.name}")

        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
            # Inject model name from folder if missing
            if "model" not in data:
                data["model"] = model_dir.name
            loaded.append(data)
            log.info(f"Loaded results: {model_dir.name}")
        except Exception as e:
            log.error(f"Failed to parse {meta_path}: {e}")

    log.info(f"Total models parsed: {len(loaded)}")
    return loaded


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Normalise into a tidy DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def results_to_dataframe(raw_results: list[dict]) -> pd.DataFrame:
    """
    Convert parsed metadata dicts into a tidy long-format DataFrame.

    Output columns:
      model      — model name (e.g. "phi-3-mini")
      task       — benchmark name (e.g. "mmlu")
      task_label — human label (e.g. "MMLU")
      metric     — metric key used (e.g. "acc")
      score      — float 0–1
      score_pct  — score * 100 (for chart axes)
      stderr     — standard error (float, 0 if not available)
      timestamp  — ISO timestamp of the run

    Returns:
        pd.DataFrame with one row per (model, task) pair.
    """
    rows = []

    for entry in raw_results:
        model_name = entry.get("model", "unknown")
        timestamp  = entry.get("timestamp", "")
        task_data  = entry.get("results", {})

        if not task_data:
            log.warning(f"No task results for model: {model_name}")
            continue

        for task, metrics in task_data.items():
            if not isinstance(metrics, dict):
                continue
            if "error" in metrics:
                log.warning(f"Skipping errored task: {model_name}/{task} — {metrics['error']}")
                continue

            metric_key = TASK_METRICS.get(task, "acc")
            score      = _extract_score(metrics, metric_key)
            stderr     = _extract_stderr(metrics, metric_key)

            if score is None:
                log.warning(f"Could not extract score for {model_name}/{task} — skipping.")
                continue

            rows.append({
                "model"      : model_name,
                "task"       : task,
                "task_label" : TASK_LABELS.get(task, task.upper()),
                "metric"     : metric_key,
                "score"      : round(score, 4),
                "score_pct"  : round(score * 100, 2),
                "stderr"     : round(stderr, 4),
                "timestamp"  : timestamp,
            })

    if not rows:
        log.error("DataFrame is empty — no valid scores were extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Consistent model ordering for charts (by mean score descending)
    model_order = (
        df.groupby("model")["score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values(["model", "task"]).reset_index(drop=True)

    log.info(f"DataFrame built: {len(df)} rows  |  models: {df['model'].unique().tolist()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Convenience pivot tables for reporter.py
# ══════════════════════════════════════════════════════════════════════════════

def to_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide-format pivot: rows = models, columns = tasks, values = score_pct.
    Useful for the heatmap and comparison tables in the report.

    Example:
                  MMLU   GSM8K  HumanEval  ARC-C
      phi-3-mini  61.2   34.1   32.9       57.8
      mistral-7b  64.5   41.3   29.1       60.2
    """
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index   = "model",
        columns = "task_label",
        values  = "score_pct",
        aggfunc = "mean",
    ).round(2)
    return pivot


def to_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table: one row per model with mean score across all tasks.
    Sorted best → worst.
    """
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby("model")["score_pct"]
        .agg(mean_score="mean", tasks_run="count")
        .round(2)
        .sort_values("mean_score", ascending=False)
        .reset_index()
    )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_score(metrics: dict, primary_key: str) -> float | None:
    """
    Try to extract a numeric score from a metrics dict.
    Falls back through common key variants before giving up.
    """
    fallback_keys = [primary_key, "acc", "exact_match", "pass@1", "acc_norm"]
    for key in fallback_keys:
        val = metrics.get(key)
        if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)
    return None


def _extract_stderr(metrics: dict, primary_key: str) -> float:
    """Extract standard error; return 0.0 if not present."""
    stderr = metrics.get(f"{primary_key}_stderr", 0.0)
    if stderr is None or np.isnan(stderr):
        return 0.0
    return float(stderr)