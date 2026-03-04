"""
display.py — CLI display commands for the SLM Evaluation Harness.

Provides:
  - cmd_list_models() : print all models registered in configs/models.yaml
  - cmd_list_tasks()  : print all tasks registered in configs/benchmarks.yaml
"""


def cmd_list_models(models_cfg: dict):
    """Print all models registered in models.yaml."""
    print(f"\n{'─' * 68}")
    print(f"  {'NAME':<25} {'HF PATH':<33} MODE")
    print(f"{'─' * 68}")
    for m in models_cfg.get("models", []):
        mode = "4-bit" if m.get("load_in_4bit") else "fp16 "
        print(f"  {m['name']:<25} {m['hf_path']:<33} [{mode}]")
    print()


def cmd_list_tasks(benchmarks_cfg: dict):
    """Print all tasks registered in benchmarks.yaml."""
    print(f"\n{'─' * 50}")
    print(f"  {'TASK':<20} {'FEW-SHOT':<12} LIMIT")
    print(f"{'─' * 50}")
    for t in benchmarks_cfg.get("tasks", []):
        limit = str(t.get("limit", "full"))
        print(f"  {t['name']:<20} {t['num_fewshot']:<12} {limit}")
    print()