"""
reporter.py — Report Generator
================================
Generates three output formats from the parsed results DataFrame:
  - PDF report  (fpdf2 + matplotlib figures)
  - HTML report (jinja2 template + inline base64 charts)
  - Dashboard   (Streamlit interactive app)

Called by main.py after parser.py builds the DataFrame.
"""

import io
import os
import base64
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/Docker
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from fpdf import FPDF
from jinja2 import Template

from .parser import to_pivot, to_summary

log = logging.getLogger(__name__)

# ── Palette ────────────────────────────────────────────────────────────────────
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
sns.set_theme(style="whitegrid", palette=COLORS)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def _fig_bar_chart(df: pd.DataFrame, figures_dir: Path) -> Path:
    """
    Grouped bar chart: one group per task, one bar per model.
    Best for comparing models side-by-side on each benchmark.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    tasks   = df["task_label"].unique()
    models  = df["model"].cat.categories.tolist() if hasattr(df["model"], "cat") else df["model"].unique().tolist()
    x       = np.arange(len(tasks))
    width   = 0.8 / len(models)

    for i, model in enumerate(models):
        sub    = df[df["model"] == model].set_index("task_label")
        scores = [sub.loc[t, "score_pct"] if t in sub.index else 0 for t in tasks]
        errs   = [sub.loc[t, "stderr"] * 100 if t in sub.index else 0 for t in tasks]
        offset = (i - len(models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, scores, width * 0.9, label=model,
                        color=COLORS[i % len(COLORS)], yerr=errs,
                        capsize=3, error_kw={"linewidth": 1})
        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Model Comparison Across Benchmarks", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    out = figures_dir / "bar_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out}")
    return out


def _fig_heatmap(df: pd.DataFrame, figures_dir: Path) -> Path:
    """
    Heatmap: rows = models, columns = tasks.
    Great for spotting where each model is weak.
    """
    pivot = to_pivot(df)
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        ax          = ax,
        annot       = True,
        fmt         = ".1f",
        cmap        = "RdYlGn",
        vmin        = 0,
        vmax        = 100,
        linewidths  = 0.5,
        cbar_kws    = {"label": "Score (%)"},
    )
    ax.set_title("Score Heatmap — Model × Benchmark", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    out = figures_dir / "heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out}")
    return out


def _fig_radar_chart(df: pd.DataFrame, figures_dir: Path) -> Path:
    """
    Radar / spider chart: one polygon per model across all tasks.
    Ideal for holistic profile comparison.
    """
    pivot  = to_pivot(df)
    if pivot.empty:
        return None

    tasks  = list(pivot.columns)
    n      = len(tasks)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for i, (model, row) in enumerate(pivot.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, "o-", linewidth=2,
                label=str(model), color=COLORS[i % len(COLORS)])
        ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels=tasks, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title("Benchmark Radar Chart", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), title="Model")
    plt.tight_layout()

    out = figures_dir / "radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out}")
    return out


def _fig_gsm8k_humaneval(df: pd.DataFrame, figures_dir: Path) -> Path:
    """
    Focused bar chart for GSM8K + HumanEval — the two hardest-to-game tasks.
    """
    focus = df[df["task"].isin(["gsm8k", "humaneval"])].copy()
    if focus.empty:
        log.warning("No GSM8K or HumanEval data found for focused chart.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data    = focus,
        x       = "task_label",
        y       = "score_pct",
        hue     = "model",
        ax      = ax,
        palette = COLORS,
        capsize = 0.05,
    )
    ax.set_title("GSM8K & HumanEval — Reasoning & Coding Focus",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    out = figures_dir / "gsm8k_humaneval.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out}")
    return out


def _generate_all_figures(df: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    """
    Generate all four figures and return their paths.
    Returns: {chart_name: Path}
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    figures = {}
    generators = {
        "bar_chart"       : _fig_bar_chart,
        "heatmap"         : _fig_heatmap,
        "radar"           : _fig_radar_chart,
        "gsm8k_humaneval" : _fig_gsm8k_humaneval,
    }

    for name, fn in generators.items():
        try:
            path = fn(df, figures_dir)
            if path:
                figures[name] = path
        except Exception as e:
            log.error(f"Figure '{name}' failed: {e}")

    return figures


def _img_to_base64(path: Path) -> str:
    """Encode an image file as base64 string for embedding in HTML."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

class EvalReport(FPDF):
    """Custom FPDF subclass with header and footer."""

    def __init__(self, title: str):
        super().__init__()
        self.report_title = title

    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(30, 55, 100)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, self.report_title, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}  |  LLM Evaluation Harness", align="C")


def generate_pdf_report(df: pd.DataFrame, output_path: str, figures_dir: Path):
    """
    Build a multi-page PDF report:
      Page 1 — Title + summary table
      Page 2 — Bar chart + heatmap
      Page 3 — Radar chart + GSM8K/HumanEval focus
      Page 4 — Analysis notes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Generating figures...")
    figures = _generate_all_figures(df, figures_dir)

    summary = to_summary(df)
    pivot   = to_pivot(df)
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf = EvalReport(title="LLM Evaluation Harness — Benchmark Report")
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: Title + Summary Table ─────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(8)
    pdf.cell(0, 12, "LLM Benchmark Evaluation Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {ts}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Subtitle
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Overall Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Summary table header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 235, 245)
    pdf.cell(70, 8, "Model",       fill=True, border=1)
    pdf.cell(50, 8, "Mean Score",  fill=True, border=1)
    pdf.cell(40, 8, "Tasks Run",   fill=True, border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    for _, row in summary.iterrows():
        pdf.cell(70, 8, str(row["model"]),             border=1)
        pdf.cell(50, 8, f"{row['mean_score']:.2f}%",   border=1)
        pdf.cell(40, 8, str(row["tasks_run"]),          border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)

    # Per-task pivot table
    if not pivot.empty:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Scores by Benchmark (%)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        col_w  = 30
        task_cols = list(pivot.columns)
        # Header row
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 235, 245)
        pdf.cell(50, 7, "Model", fill=True, border=1)
        for col in task_cols:
            pdf.cell(col_w, 7, str(col), fill=True, border=1, align="C")
        pdf.ln()

        # Data rows
        pdf.set_font("Helvetica", "", 9)
        for model, row in pivot.iterrows():
            pdf.cell(50, 7, str(model), border=1)
            for col in task_cols:
                val = row.get(col, None)
                txt = f"{val:.1f}" if val is not None and not np.isnan(val) else "—"
                pdf.cell(col_w, 7, txt, border=1, align="C")
            pdf.ln()

    # ── Page 2: Bar chart + Heatmap ───────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.ln(4)

    if "bar_chart" in figures:
        pdf.cell(0, 8, "Model Comparison — All Benchmarks", new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(figures["bar_chart"]), x=10, w=190)
        pdf.ln(6)

    if "heatmap" in figures:
        pdf.cell(0, 8, "Score Heatmap", new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(figures["heatmap"]), x=10, w=190)

    # ── Page 3: Radar + GSM8K/HumanEval ──────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.ln(4)

    if "radar" in figures:
        pdf.cell(0, 8, "Radar Chart — Model Profiles", new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(figures["radar"]), x=30, w=145)
        pdf.ln(6)

    if "gsm8k_humaneval" in figures:
        pdf.cell(0, 8, "GSM8K & HumanEval Focus", new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(figures["gsm8k_humaneval"]), x=10, w=190)

    # ── Page 4: Analysis notes ────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.ln(4)
    pdf.cell(0, 10, "Analysis Notes", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)

    best_model = summary.iloc[0]["model"] if not summary.empty else "N/A"
    best_score = summary.iloc[0]["mean_score"] if not summary.empty else 0

    analysis_lines = [
        f"Best overall model: {best_model} ({best_score:.1f}% mean across benchmarks).",
        "",
        "MMLU tests broad world knowledge across 57 subjects. Small models typically",
        "plateau around 45-65% — random baseline is 25%.",
        "",
        "GSM8K measures multi-step mathematical reasoning. This is where smaller",
        "models drop off sharply — fewer parameters means less working memory",
        "for chained calculations.",
        "",
        "HumanEval (pass@1) tests functional code generation. Even 7B models",
        "struggle here — GPT-4 level performance requires much larger models.",
        "",
        "ARC Challenge uses science exam questions that require reasoning beyond",
        "pattern matching. Models below 3B often score near random (33%).",
        "",
        "Key takeaway: parameter count is not the only driver — training data quality,",
        "instruction tuning, and RLHF alignment all significantly affect benchmark",
        "performance. Phi-3 Mini and Qwen2 punch above their size class due to",
        "high-quality synthetic training data.",
    ]

    for line in analysis_lines:
        pdf.multi_cell(0, 6, line)

    pdf.output(str(output_path))
    log.info(f"PDF report written → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LLM Evaluation Report</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: "Segoe UI", sans-serif; background: #f4f6fa; color: #222; }
    header { background: #1e3764; color: white; padding: 28px 40px; }
    header h1 { font-size: 1.8rem; }
    header p  { opacity: .7; margin-top: 4px; }
    main  { max-width: 1100px; margin: 30px auto; padding: 0 20px; }
    h2    { margin: 30px 0 12px; font-size: 1.3rem; color: #1e3764; border-bottom: 2px solid #dde3f0; padding-bottom: 6px; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    th    { background: #1e3764; color: white; padding: 10px 14px; font-size: .85rem; }
    td    { padding: 9px 14px; border-bottom: 1px solid #eee; font-size: .88rem; }
    tr:last-child td { border: none; }
    tr:hover td      { background: #f0f4ff; }
    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 10px; }
    .chart-card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    .chart-card img  { width: 100%; height: auto; border-radius: 6px; }
    .chart-card h3   { font-size: .95rem; margin-bottom: 10px; color: #333; }
    footer { text-align: center; padding: 24px; color: #999; font-size: .8rem; margin-top: 40px; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: .8rem; font-weight: 600; }
    .good  { background: #d4edda; color: #155724; }
    .mid   { background: #fff3cd; color: #856404; }
    .low   { background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
<header>
  <h1>LLM Benchmark Evaluation Report</h1>
  <p>Generated: {{ timestamp }} &nbsp;|&nbsp; Models evaluated: {{ n_models }}</p>
</header>
<main>

  <h2>Overall Summary</h2>
  <table>
    <tr>{% for col in summary_cols %}<th>{{ col }}</th>{% endfor %}</tr>
    {% for row in summary_rows %}
    <tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
    {% endfor %}
  </table>

  <h2>Scores by Benchmark</h2>
  <table>
    <tr>
      <th>Model</th>
      {% for task in task_cols %}<th>{{ task }}</th>{% endfor %}
    </tr>
    {% for row in pivot_rows %}
    <tr>
      <td><strong>{{ row.model }}</strong></td>
      {% for v in row.scores %}
      <td>
        {% if v != '—' %}
        <span class="badge {% if v|float >= 60 %}good{% elif v|float >= 40 %}mid{% else %}low{% endif %}">
          {{ v }}%
        </span>
        {% else %}—{% endif %}
      </td>
      {% endfor %}
    </tr>
    {% endfor %}
  </table>

  <h2>Charts</h2>
  <div class="chart-grid">
    {% for chart in charts %}
    <div class="chart-card">
      <h3>{{ chart.title }}</h3>
      <img src="data:image/png;base64,{{ chart.b64 }}" alt="{{ chart.title }}">
    </div>
    {% endfor %}
  </div>

</main>
<footer>LLM Evaluation Harness — Benchmarks: MMLU · GSM8K · HumanEval · ARC-C · GPQA</footer>
</body>
</html>
"""

def generate_html_report(df: pd.DataFrame, output_path: str, figures_dir: Path):
    """
    Render a self-contained HTML report with inline base64 charts.
    No external dependencies — open it in any browser.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures = _generate_all_figures(df, figures_dir)
    summary = to_summary(df)
    pivot   = to_pivot(df)

    # Build summary table data
    summary_cols = ["Model", "Mean Score (%)", "Tasks Run"]
    summary_rows = [
        [row["model"], f"{row['mean_score']:.2f}", str(row["tasks_run"])]
        for _, row in summary.iterrows()
    ]

    # Build pivot table data
    task_cols  = list(pivot.columns) if not pivot.empty else []
    pivot_rows = []
    if not pivot.empty:
        for model, row in pivot.iterrows():
            scores = [
                f"{row[t]:.1f}" if not np.isnan(row[t]) else "—"
                for t in task_cols
            ]
            pivot_rows.append({"model": model, "scores": scores})

    # Encode charts
    chart_titles = {
        "bar_chart"       : "Model Comparison — All Benchmarks",
        "heatmap"         : "Score Heatmap",
        "radar"           : "Radar Chart",
        "gsm8k_humaneval" : "GSM8K & HumanEval Focus",
    }
    charts = []
    for key, title in chart_titles.items():
        if key in figures:
            charts.append({
                "title" : title,
                "b64"   : _img_to_base64(figures[key]),
            })

    html = Template(HTML_TEMPLATE).render(
        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_models    = df["model"].nunique(),
        summary_cols= summary_cols,
        summary_rows= summary_rows,
        task_cols   = task_cols,
        pivot_rows  = pivot_rows,
        charts      = charts,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"HTML report written → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_SCRIPT = """
import sys, json, pickle
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title = "LLM Eval Dashboard",
    layout     = "wide",
    page_icon  = "🤖",
)

@st.cache_data
def load_data():
    from parser import parse_results, results_to_dataframe, to_pivot, to_summary
    raw = parse_results(Path("benchmarks/results"))
    return results_to_dataframe(raw)

df = load_data()

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🤖 LLM Benchmark Evaluation Dashboard")
st.caption(f"Benchmarks: MMLU · GSM8K · HumanEval · ARC-C · GPQA  |  Models: {df['model'].nunique()}")

# ── Sidebar filters ────────────────────────────────────────────────────────
st.sidebar.header("Filters")
all_models = df["model"].unique().tolist()
sel_models = st.sidebar.multiselect("Models", all_models, default=all_models)
all_tasks  = df["task_label"].unique().tolist()
sel_tasks  = st.sidebar.multiselect("Tasks", all_tasks, default=all_tasks)

dff = df[df["model"].isin(sel_models) & df["task_label"].isin(sel_tasks)]

# ── KPI row ────────────────────────────────────────────────────────────────
cols = st.columns(len(sel_models))
for i, model in enumerate(sel_models):
    mean = dff[dff["model"] == model]["score_pct"].mean()
    cols[i].metric(label=str(model), value=f"{mean:.1f}%", delta="mean score")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Score by Benchmark")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=dff, x="task_label", y="score_pct", hue="model", ax=ax)
    ax.set_ylabel("Score (%)"); ax.set_xlabel(""); ax.set_ylim(0, 105)
    ax.legend(title="Model", fontsize=7)
    ax.grid(axis="y", alpha=0.4)
    st.pyplot(fig)

with c2:
    st.subheader("Heatmap")
    pivot = dff.pivot_table(index="model", columns="task_label", values="score_pct", aggfunc="mean")
    fig2, ax2 = plt.subplots(figsize=(7, max(3, len(pivot) * 0.7)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100, ax=ax2)
    ax2.set_ylabel(""); ax2.set_xlabel("")
    st.pyplot(fig2)

# ── Data table ─────────────────────────────────────────────────────────────
st.subheader("Full Results Table")
st.dataframe(
    dff[["model", "task_label", "score_pct", "stderr", "metric"]]
    .rename(columns={"task_label": "task", "score_pct": "score (%)", "stderr": "±stderr"})
    .sort_values(["model", "task"])
    .reset_index(drop=True),
    use_container_width=True,
)
"""

def generate_dashboard(df: pd.DataFrame):
    """
    Write a standalone Streamlit app script and launch it.
    Called by main.py when --format dashboard is set.
    """
    script_path = Path("reports") / "dashboard_app.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(DASHBOARD_SCRIPT)
    log.info(f"Dashboard script written → {script_path}")

    log.info("Launching Streamlit...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script_path),
         "--server.headless", "true"],
        check=True,
    )