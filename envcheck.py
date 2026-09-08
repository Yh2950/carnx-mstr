"""
Fail early with a helpful message when a script is run with the wrong Python.

Every entry-point script does ``import envcheck`` on its first line. If torch /
pandas / numpy / yfinance are missing, the user almost certainly ran the system
``python3`` instead of the project venv.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_REQUIRED = ("numpy", "pandas", "torch", "yfinance")


def _missing() -> list[str]:
    return [m for m in _REQUIRED if importlib.util.find_spec(m) is None]


def _under_streamlit() -> bool:
    # running inside `streamlit run` (local or Streamlit Community Cloud)
    return "streamlit" in sys.modules or bool(
        os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("STREAMLIT_RUNTIME_ENV")
    )


def ensure() -> None:
    miss = _missing()
    if not miss:
        return
    if _under_streamlit():
        # a bare sys.exit() here just yields Streamlit's opaque "Error running
        # app".  Let the real ``import torch`` fail a few lines later with a
        # traceback the deploy logs can show.
        return
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(here, ".venv", "bin", "python")
    msg = [
        "",
        "  ✗ wrong Python interpreter — missing: " + ", ".join(miss),
        f"    you ran: {sys.executable}",
        "",
        "  run it with the project venv instead:",
        f"    {venv_py} {' '.join(sys.argv) or '<script>.py'}",
        "",
        "  or activate the venv once for this terminal:",
        f"    source {os.path.join(here, '.venv', 'bin', 'activate')}",
        "",
        "  (in PyCharm: Settings → Project → Python Interpreter → select .venv/bin/python)",
        "",
    ]
    sys.exit("\n".join(msg))


ensure()
