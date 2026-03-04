"""
cli.py — Argument parser for the SLM Evaluation Harness.

Provides:
  - build_parser() : construct and return the ArgumentParser
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="SLM Evaluation Harness — evaluate small SLMs on MMLU, GSM8K, HumanEval, ARC, GPQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --eval --model phi-3-mini
  python main.py --eval --all-models
  python main.py --report --format pdf
  python main.py --report --format html
  python main.py --report --format dashboard
  python main.py --all --model mistral-7b --format pdf
  python main.py --all --all-models
  python main.py --list-models
  python main.py --list-tasks
  python main.py --dry-run --model gemma-2b
        """
    )

    # ── Actions ────────────────────────────────────────────────────────────────
    actions = p.add_argument_group("Actions")
    actions.add_argument(
        "--eval",        action="store_true",
        help="Run benchmark evaluations (requires --model or --all-models)"
    )
    actions.add_argument(
        "--report",      action="store_true",
        help="Generate a report from existing results"
    )
    actions.add_argument(
        "--all",         action="store_true",
        help="Run --eval then --report in one command"
    )
    actions.add_argument(
        "--list-models", action="store_true",
        help="Print all models registered in configs/models.yaml"
    )
    actions.add_argument(
        "--list-tasks",  action="store_true",
        help="Print all benchmark tasks in configs/benchmarks.yaml"
    )

    # ── Model selection ────────────────────────────────────────────────────────
    model_grp = p.add_argument_group("Model selection")
    model_sel = model_grp.add_mutually_exclusive_group()
    model_sel.add_argument(
        "--model",      type=str, metavar="NAME",
        help="Name of a single model to evaluate (must exist in configs/models.yaml)"
    )
    model_sel.add_argument(
        "--all-models", action="store_true",
        help="Evaluate all models defined in configs/models.yaml"
    )

    # ── Report options ─────────────────────────────────────────────────────────
    report_grp = p.add_argument_group("Report options")
    report_grp.add_argument(
        "--format", type=str, default="pdf",
        choices=["pdf", "html", "dashboard"],
        help="Report output format (default: pdf)"
    )
    report_grp.add_argument(
        "--output", type=str, default=None,
        help="Custom file path for pdf/html output (default: reports/report_<timestamp>)"
    )

    # ── Dev / debug ────────────────────────────────────────────────────────────
    dev_grp = p.add_argument_group("Dev options")
    dev_grp.add_argument(
        "--dry-run", action="store_true",
        help="Validate configs and resolve model paths without running evaluation"
    )
    dev_grp.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG level logging"
    )

    return p