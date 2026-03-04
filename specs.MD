# Small lenguje models harness evalation with standard Benchmarks
this project aims to create a simple model benchmark evaluation for small lenguaje models (SLMs) againts industry-standard benchmarks-MMLU,GSM8K, HumalEval, ARC challenge and GPQA to produce a polished PDF report and interative streamlit dashboard.
## Goals
| # | Goal | Priority |
|---|------|----------|
| G1 | Run 3–7 small models against 5 standard benchmarks on a local RTX 3050 | Must have |
| G2 | Generate a professional PDF report comparing model performance | Must have |
| G3 | Launch an interactive Streamlit dashboard with filters and charts | Must have |
| G4 | Support 4-bit quantization for 7B models to fit in 8GB VRAM | Must have |
| G5 | Full reproducibility via Docker | Should have |
| G6 | Fallback to custom eval scripts if lm-evaluation-harness fails | Should have |
| G7 | Makefile one-command interface for demos | Should have |

### Non-Goals

- No cloud deployment or hosted service
- No real-time / streaming inference
- No fine-tuning or model training
- No support for proprietary APIs (OpenAI, Anthropic) in v1
- No multi-GPU support (single RTX 3050 target)

---


## 4. Target Audience

**Primary:** Technical interviewers and porfolio for practice
**Secondary:** The developer — this is a personal tool for quickly comparing
new models as they are released.

---

## Requirements

## Technology Stack

### Core Stack

| Layer | Tool | Version | Reason |
|-------|------|---------|--------|
| Language | Python | 3.11 | Latest stable with good HF support |
| Eval framework | lm-evaluation-harness | latest | Industry standard, used by HF Leaderboard |
| Model loading | transformers + accelerate | latest | HuggingFace ecosystem |
| Quantization | bitsandbytes | latest | 4-bit NF4 for 7B models on 8GB VRAM |
| Data | pandas + numpy | latest | Result parsing and aggregation |
| Visualisation | matplotlib + seaborn | latest | Charts for PDF and dashboard |
| PDF generation | fpdf2 | latest | Pure Python, no LaTeX dependency |
| HTML templating | jinja2 | latest | HTML report rendering |
| Dashboard | Streamlit | latest | Fastest path to interactive UI |
| Config | PyYAML | latest | models.yaml + benchmarks.yaml |
| Containerisation | Docker + Compose | latest | Reproducibility |
| Task runner | Makefile | — | One-command interface for demos |

###  Python Environment

```
Virtual environment name : slm
Python version           : 3.11
CUDA target              : 11.8 (RTX 3050 compatible)
Activation               : source slm/bin/activate
Setup script             : setup_env.sh
```

### Models Targeted (v1)

| Name | HF Path | VRAM mode |
|------|---------|-----------|
| TinyLlama 1.1B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | fp16 |
| Phi-3 Mini 3.8B | microsoft/Phi-3-mini-4k-instruct | fp16 |
| Gemma 2B IT | google/gemma-2b-it | fp16 |
| Mistral 7B Instruct | mistralai/Mistral-7B-Instruct-v0.2 | 4-bit |
| Qwen2 1.5B | Qwen/Qwen2-1.5B-Instruct | fp16 |
| StableLM 2 1.6B | stabilityai/stablelm-2-1_6b-chat | fp16 |
| OPT 2.7B | facebook/opt-2.7b | fp16 |

---

### Benchmarks

| Benchmark | Task key | Few-shot | Dev limit | Full limit | Primary metric |
|-----------|----------|----------|-----------|------------|----------------|
| MMLU | mmlu | 5 | 100 | 14,042 | acc |
| GSM8K | gsm8k | 8 | 50 | 1,319 | exact_match |
| HumanEval | humaneval | 0 | 50 | 164 | pass@1 |
| ARC Challenge | arc_challenge | 25 | 100 | 1,172 | acc_norm |
| GPQA | gpqa | 0 | 50 | 448 | acc |

> **Dev limit** is used during development to keep iteration fast.  
> **Full limit** (`None` in config) is used for final portfolio-grade results.

---


## Architecture

```
configs/
  models.yaml          ← model registry
  benchmarks.yaml      ← tasks, shots, limits

main.py                ← CLI entry point (argparse)
  │
  ├── loader.py        ← load_model() / unload_model()
  │     └── HuggingFace AutoModelForCausalLM
  │         BitsAndBytesConfig (4-bit NF4)
  │
  ├── runner.py        ← run_evaluation()
  │     └── lm_eval.simple_evaluate()
  │         HFLM adapter wraps loaded model
  │         writes raw JSON → benchmarks/results/<model>/
  │
  ├── parser.py        ← parse_results() / results_to_dataframe()
  │     └── reads metadata.json files
  │         normalises metrics into tidy DataFrame
  │         to_pivot() / to_summary() helpers
  │
  └── reporter.py      ← generate_pdf_report()
        │               generate_html_report()
        │               generate_dashboard()
        └── 4 matplotlib figures
            fpdf2 PDF (4 pages)
            Jinja2 HTML (self-contained)
            Streamlit app script (launched via subprocess)
```

---

## CLI Interface

```bash
# Evaluate a single model
python main.py --eval --model phi-3-mini

# Evaluate all models
python main.py --eval --all-models

# Generate PDF report from existing results
python main.py --report --format pdf

# Launch Streamlit dashboard
python main.py --report --format dashboard

# Full pipeline: eval + report in one command
python main.py --all --all-models --format pdf

# Dry run (validate configs only)
python main.py --dry-run --model mistral-7b

# List registered models
python main.py --list-models

# List configured tasks
python main.py --list-tasks
```

---

## project breakdown

## project structure 


## 13. Development Phases

###  Core pipeline (sprint 1)
- [ ] `setup_env.sh` working on RTX 3050
- [ ] `loader.py` — load + unload with VRAM logging
- [ ] `runner.py` — lm_eval integration, per-task limits
- [ ] `parser.py` — JSON → DataFrame, pivot, summary
- [ ] Dev-mode test: TinyLlama + MMLU limit=50

### Phase 2 — Reports (sprint 2)
- [ ] `reporter.py` — 4 matplotlib figures
- [ ] PDF report — 4 pages with fpdf2
- [ ] Streamlit dashboard with filters
- [ ] HTML report (self-contained)
- [ ] Test with 3 models, all 5 tasks, dev limits

### Phase 3 — Polish + Portfolio (sprint 3)
- [ ] Full run: all 7 models, full limits (overnight)
- [ ] `runner_custom.py` — backup MMLU + GSM8K scripts
- [ ] Docker setup — Dockerfile + compose
- [ ] `tests/` — unit tests for parser and loader
- [ ] README with GIF demo, badges, results table
- [ ] GitHub release with PDF report attached

---
