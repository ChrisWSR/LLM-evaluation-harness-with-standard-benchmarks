"""
utils.py — Shared utility functions for the SLM Evaluation Harness.

Provides:
  - ensure_dirs()         : create required project directories
  - load_yaml()           : load a YAML config file safely
  - save_run_metadata()   : persist a JSON sidecar next to raw results
  - print_banner()        : print the ASCII banner on startup
  - print_summary_table() : print a per-model score table to stdout
"""

import json
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path

# utils/utils.py
from config import LOGS_DIR, RESULTS_DIR, REPORTS_DIR, FIGURES_DIR
log = logging.getLogger(__name__)


# ── Directory bootstrap ────────────────────────────────────────────────────────

def ensure_dirs():
    """Create all required project directories."""
    for d in [RESULTS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log.info("Project directories verified.")


# ── YAML loader ────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    """Load and return a YAML config file, exiting on failure."""
    if not path.exists():
        log.error(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Metadata persistence ───────────────────────────────────────────────────────

def save_run_metadata(model_name: str, tasks: list, results: dict, elapsed: float):
    """Save a JSON sidecar file next to each model's raw results."""
    meta = {
        "model"       : model_name,
        "tasks"       : tasks,
        "timestamp"   : datetime.utcnow().isoformat(),
        "elapsed_sec" : round(elapsed, 2),
        "results"     : results,
    }
    out_path = RESULTS_DIR / model_name / "metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved → {out_path}")


# ── Console helpers ────────────────────────────────────────────────────────────

def print_banner():
    """Print the startup ASCII banner."""
    print("""
╔══════════════════════════════════════════════════╗
║         SLM Evaluation Harness  v0.9             ║
╚══════════════════════════════════════════════════╝
""")


# Primary metric to display per benchmark task
_METRIC_MAP = {
    "mmlu"          : "acc",
    "gsm8k"         : "exact_match",
    "humaneval"     : "pass@1",
    "arc_challenge" : "acc_norm",
    "gpqa"          : "acc",
}


def print_summary_table(results: dict, model_name: str):
    """Print a quick score summary to the console after each model run."""
    print(f"\n{'─' * 55}")
    print(f"  ✓ Results: {model_name}")
    print(f"{'─' * 55}")

    task_results = results.get("results", {})
    if not task_results:
        print("  No results found.\n")
        return

    for task, metrics in task_results.items():
        metric_key = _METRIC_MAP.get(task, "acc")
        score      = metrics.get(metric_key, metrics.get("acc", 0.0)) or 0.0
        stderr     = metrics.get(f"{metric_key}_stderr", 0.0) or 0.0
        bar        = "█" * int(score * 20)
        print(f"  {task:<20} {score * 100:5.1f}%  ±{stderr * 100:.1f}  {bar}")

    print(f"{'─' * 55}\n")