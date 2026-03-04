#!/usr/bin/env python3
"""
SLM Evaluation Harness — Main Entry Point
==========================================
Integrates: config.py | utils.py | display.py | cli.py | commands.py
            loader.py | runner.py | parser.py  | reporter.py

Usage examples:
  python main.py --eval --model phi-3-mini
  python main.py --eval --all-models
  python main.py --report
  python main.py --report --format pdf
  python main.py --report --format dashboard
  python main.py --all --model mistral-7b
  python main.py --all --all-models
  python main.py --list-models
  python main.py --list-tasks
"""

import logging
import sys

# main.py
from config          import CONFIG_MODELS, CONFIG_BENCHMARKS, LOGS_DIR
from utils.utils     import ensure_dirs, load_yaml, print_banner
from cli.args        import build_parser
from cli.display     import cmd_list_models, cmd_list_tasks
from cli.commands    import resolve_models, cmd_eval, cmd_report
from cli import build_parser, cmd_list_models, cmd_list_tasks, resolve_models, cmd_eval, cmd_report
# ── Logging setup ──────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "run.log"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()
    print_banner()

    parser = build_parser()
    args   = parser.parse_args()

    # No arguments → show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Verbose / DEBUG logging enabled.")

    # Load configs (needed by most commands)
    models_cfg     = load_yaml(CONFIG_MODELS)
    benchmarks_cfg = load_yaml(CONFIG_BENCHMARKS)

    # ── Route to command ───────────────────────────────────────────────────────

    if args.list_models:
        cmd_list_models(models_cfg)
        sys.exit(0)

    if args.list_tasks:
        cmd_list_tasks(benchmarks_cfg)
        sys.exit(0)

    if args.dry_run:
        log.info("── Dry run: validating configs only ──")
        models = resolve_models(args, models_cfg)
        log.info(f"  Models  : {[m['name'] for m in models]}")
        log.info(f"  Tasks   : {[t['name'] for t in benchmarks_cfg['tasks']]}")
        log.info(f"  Format  : {args.format}")
        log.info("Dry run complete. Nothing was evaluated or written.")
        sys.exit(0)

    if args.eval or args.all:
        cmd_eval(args, models_cfg, benchmarks_cfg)

    if args.report or args.all:
        cmd_report(args)

    # If somehow nothing matched
    if not any([
        args.eval, args.report, args.all,
        args.list_models, args.list_tasks, args.dry_run
    ]):
        parser.print_help()


if __name__ == "__main__":
    main()