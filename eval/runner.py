"""
runner.py — Benchmark Runner
=============================
Wraps EleutherAI's lm-evaluation-harness to run MMLU, GSM8K,
HumanEval, ARC Challenge and GPQA against a loaded HF model.

Flow:
  1. Wraps the already-loaded model in lm_eval's HFLM adapter
  2. Calls lm_eval.simple_evaluate() with per-task shot/limit settings
  3. Saves raw JSON output to benchmarks/results/<model_name>/
  4. Returns the results dict back to main.py
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import lm_eval
from lm_eval.models.huggingface import HFLM

log = logging.getLogger(__name__)

# Tasks we support and the metric that matters for each
TASK_METRICS = {
    "mmlu"          : "acc",
    "gsm8k"         : "exact_match",
    "humaneval"     : "pass@1",
    "arc_challenge" : "acc_norm",
    "gpqa"          : "acc",
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model,
    tokenizer,
    model_cfg   : dict,
    tasks       : list[str],
    num_fewshot : dict[str, int],
    limits      : dict[str, int | None],
    results_dir : Path,
) -> dict:
    """
    Run all benchmark tasks for a single model.

    Args:
        model       : loaded HF CausalLM (from loader.py)
        tokenizer   : matching HF tokenizer
        model_cfg   : full config dict for this model (name, hf_path, device, …)
        tasks       : list of task names, e.g. ["mmlu", "gsm8k"]
        num_fewshot : {task_name: n_shots}, e.g. {"mmlu": 5, "gsm8k": 8}
        limits      : {task_name: int|None} — None means full dataset
        results_dir : Path to benchmarks/results/

    Returns:
        Raw results dict from lm_eval.simple_evaluate()
    """
    model_name = model_cfg["name"]
    device     = model_cfg.get("device", "cuda")

    log.info(f"Wrapping model in HFLM adapter: {model_name}")
    lm_model = _wrap_model(model, tokenizer, device)

    all_results = {}

    # Run each task individually so we can apply per-task limits/shots
    for task in tasks:
        shots = num_fewshot.get(task, 0)
        limit = limits.get(task, None)

        log.info(f"  ── Task: {task}  |  few-shot={shots}  |  limit={limit or 'full'}")

        try:
            result = lm_eval.simple_evaluate(
                model        = lm_model,
                tasks        = [task],
                num_fewshot  = shots,
                limit        = limit,
                log_samples  = False,      # don't save every sample — saves disk space
                verbosity    = "WARNING",  # suppress lm_eval's own INFO spam
            )
            # Merge into combined results dict
            all_results[task] = result["results"].get(task, {})
            _log_task_score(task, all_results[task])

        except Exception as e:
            log.error(f"  Task '{task}' failed: {e}")
            all_results[task] = {"error": str(e)}

    # Wrap in lm_eval-style envelope
    combined = {"results": all_results}

    # Persist to disk
    _save_raw_results(model_name, combined, results_dir)

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _wrap_model(model, tokenizer, device: str) -> HFLM:
    """
    Wrap a loaded HuggingFace model in lm_eval's HFLM adapter.
    This lets simple_evaluate() call the model without re-downloading it.
    """
    return HFLM(
        pretrained  = model,
        tokenizer   = tokenizer,
        device      = device,
        batch_size  = "auto",   # lm_eval will find the largest batch that fits in VRAM
    )


def _log_task_score(task: str, metrics: dict) -> None:
    """Log the primary metric score for a completed task."""
    if "error" in metrics:
        log.warning(f"  {task}: ERROR — {metrics['error']}")
        return
    primary = TASK_METRICS.get(task, "acc")
    score   = metrics.get(primary, metrics.get("acc", None))
    if score is not None:
        log.info(f"  {task}: {primary} = {score * 100:.2f}%")
    else:
        log.warning(f"  {task}: metric '{primary}' not found in results.")


def _save_raw_results(model_name: str, results: dict, results_dir: Path) -> None:
    """
    Write raw JSON results to:
      benchmarks/results/<model_name>/raw_<timestamp>.json
    """
    out_dir = results_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"raw_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Raw results saved → {out_path}")


def get_available_tasks() -> list[str]:
    """Return tasks we have configured (not the full lm_eval catalogue)."""
    return list(TASK_METRICS.keys())