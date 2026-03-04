# cli/__init__.py  — optional shortcut exports
from .args     import build_parser
from .display  import cmd_list_models, cmd_list_tasks
from .commands import resolve_models, cmd_eval, cmd_report