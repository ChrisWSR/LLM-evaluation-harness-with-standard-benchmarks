"""
config.py — Central path and directory constants for the SLM Evaluation Harness.
"""

from pathlib import Path

# ── Config files ───────────────────────────────────────────────────────────────
CONFIG_MODELS     = Path("configs/models.yaml")
CONFIG_BENCHMARKS = Path("configs/benchmarks.yaml")

# ── Output directories ─────────────────────────────────────────────────────────
RESULTS_DIR = Path("benchmarks/results")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR    = Path("logs")