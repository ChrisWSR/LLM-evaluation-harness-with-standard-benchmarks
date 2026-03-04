# reporting/__init__.py
from .parser   import parse_results, results_to_dataframe
from .reporter import generate_pdf_report, generate_dashboard, generate_html_report