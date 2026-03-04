"""
commands.py — Core command handlers for the SLM Evaluation Harness.

Provides:
  - resolve_models() : resolve the list of model configs from CLI args
  - cmd_eval()       : evaluation loop  (loader → runner → summary → metadata)
  - cmd_report()     : report pipeline  (parser → reporter)
"""

import logging
import sys
import time
from datetime import datetime

# cli/commands.py
from config            import RESULTS_DIR, REPORTS_DIR, FIGURES_DIR
from utils.utils       import print_summary_table, save_run_metadata
from eval.loader       import load_model, unload_model
from eval.runner       import run_evaluation
from reporting.parser  import parse_results, results_to_dataframe
from reporting.reporter import generate_pdf_report, generate_dashboard, generate_html_report

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def resolve_models(args, models_cfg: dict) -> list[dict]:
    """
    Return list of model configs to evaluate.
      --model <name>  → single model matched by name from models.yaml
      --all-models    → every model in models.yaml
    """
    all_models = models_cfg.get("models", [])

    if not all_models:
        log.error("No models defined in configs/models.yaml.")
        sys.exit(1)

    if args.all_models:
        log.info(f"Queued {len(all_models)} model(s) for evaluation.")
        return all_models

    if args.model:
        match = [m for m in all_models if m["name"] == args.model]
        if not match:
            available = [m["name"] for m in all_models]
            log.error(f"Model '{args.model}' not found in config.")
            log.error(f"Available models: {available}")
            sys.exit(1)
        return match

    log.error("Specify --model <name> or --all-models.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# EVAL COMMAND
# ══════════════════════════════════════════════════════════════════════════════

def cmd_eval(args, models_cfg: dict, benchmarks_cfg: dict):
    """
    Main evaluation loop. For each model:

      1. loader.load_model()       → loads HF model + tokenizer onto GPU
      2. runner.run_evaluation()   → runs lm-eval tasks, writes raw JSON to disk
      3. print_summary_table()     → quick console preview
      4. save_run_metadata()       → traceability JSON sidecar
      5. loader.unload_model()     → releases VRAM before loading next model
    """
    models  = resolve_models(args, models_cfg)
    tasks   = [t["name"]                   for t in benchmarks_cfg["tasks"]]
    shots   = {t["name"]: t["num_fewshot"] for t in benchmarks_cfg["tasks"]}
    limits  = {t["name"]: t.get("limit")   for t in benchmarks_cfg["tasks"]}

    total  = len(models)
    failed = []

    log.info(f"Starting evaluation: {total} model(s) × {len(tasks)} task(s)")
    log.info(f"Tasks queued: {tasks}")

    for idx, model_cfg in enumerate(models, start=1):
        model_name = model_cfg["name"]
        hf_path    = model_cfg["hf_path"]

        log.info(f"[{idx}/{total}] ── Loading: {model_name}  ({hf_path})")

        # ── Step 1: Load via loader.py ─────────────────────────────────────────
        try:
            model, tokenizer = load_model(model_cfg)
            log.info(f"[{idx}/{total}] Model loaded onto device.")
        except Exception as e:
            log.error(f"[{idx}/{total}] Load failed for {model_name}: {e}")
            failed.append(model_name)
            continue

        # ── Step 2: Evaluate via runner.py ────────────────────────────────────
        start = time.time()
        try:
            results = run_evaluation(
                model       = model,
                tokenizer   = tokenizer,
                model_cfg   = model_cfg,
                tasks       = tasks,
                num_fewshot = shots,
                limits      = limits,
                results_dir = RESULTS_DIR,
            )
            elapsed = time.time() - start
            log.info(f"[{idx}/{total}] Eval done in {elapsed:.1f}s")

            # ── Step 3: Console summary ────────────────────────────────────────
            print_summary_table(results, model_name)

            # ── Step 4: Persist metadata ───────────────────────────────────────
            save_run_metadata(model_name, tasks, results, elapsed)

        except Exception as e:
            elapsed = time.time() - start
            log.error(f"[{idx}/{total}] Eval failed for {model_name} after {elapsed:.1f}s: {e}")
            failed.append(model_name)

        finally:
            # ── Step 5: Always free VRAM ───────────────────────────────────────
            log.info(f"[{idx}/{total}] Unloading {model_name} from memory...")
            unload_model(model)

    # ── Final summary ──────────────────────────────────────────────────────────
    succeeded = total - len(failed)
    print(f"\n{'═' * 55}")
    print(f"  Evaluation complete  →  {succeeded}/{total} models succeeded")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'═' * 55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT COMMAND
# ══════════════════════════════════════════════════════════════════════════════

def cmd_report(args):
    """
    Build reports from results already on disk. Pipeline:

      1. parser.parse_results()        → reads all metadata.json files in RESULTS_DIR
      2. parser.results_to_dataframe() → normalises scores into a tidy pandas DataFrame
      3. reporter.generate_*()         → renders PDF / HTML / Streamlit dashboard
    """
    report_format = getattr(args, "format", "pdf")
    output_path   = getattr(args, "output", None)

    log.info(f"Generating report  →  format={report_format}")
    log.info(f"Scanning results in: {RESULTS_DIR}")

    # ── Step 1 & 2: Parse via parser.py ───────────────────────────────────────
    try:
        raw_results = parse_results(RESULTS_DIR)

        if not raw_results:
            log.error("No results found in benchmarks/results/. Run --eval first.")
            sys.exit(1)

        df = results_to_dataframe(raw_results)
        log.info(f"Parsed {len(raw_results)} model result(s).")
        log.info(f"DataFrame shape: {df.shape}  |  columns: {list(df.columns)}")

    except Exception as e:
        log.error(f"Parsing failed: {e}")
        sys.exit(1)

    # ── Step 3: Render via reporter.py ────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    try:
        if report_format == "pdf":
            out = output_path or str(REPORTS_DIR / f"report_{timestamp}.pdf")
            generate_pdf_report(df=df, output_path=out, figures_dir=FIGURES_DIR)
            log.info(f"PDF report saved → {out}")
            print(f"\n  ✓ PDF report: {out}\n")

        elif report_format == "html":
            out = output_path or str(REPORTS_DIR / f"report_{timestamp}.html")
            generate_html_report(df=df, output_path=out, figures_dir=FIGURES_DIR)
            log.info(f"HTML report saved → {out}")
            print(f"\n  ✓ HTML report: {out}\n")

        elif report_format == "dashboard":
            log.info("Launching Streamlit dashboard...")
            print("\n  Launching interactive dashboard at http://localhost:8501\n")
            generate_dashboard(df=df)

        else:
            log.error(f"Unknown format '{report_format}'. Choose: pdf | html | dashboard")
            sys.exit(1)

    except Exception as e:
        log.error(f"Report generation failed: {e}")
        sys.exit(1)