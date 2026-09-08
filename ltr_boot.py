"""
CARN-X  --  force LTR widget geometry  (presentation only, no app logic)
======================================================================
Streamlit's slider / select-slider are built on ``react-aria``, which picks its
reading direction from ``navigator.language`` at first render.  On a Hebrew or
Arabic browser that makes every slider **mirrored**: the track fills from the
right and dragging the handle to the *right* lowers the value.  The app's Hebrew
*text* is unaffected -- that comes from the browser's own Unicode bidi handling,
not from ``navigator.language``.

The only reliable cure is to pin the locale *before* the frontend bundle runs,
so this module rewrites Streamlit's static ``index.html`` to add one tiny
``<script>`` at the very top of ``<head>``.  It is:

* idempotent  -- a marker comment guards against double-patching;
* self-healing -- re-applied on every process start, so a ``pip install -U
  streamlit`` that ships a fresh ``index.html`` can't silently undo it;
* cheap       -- a single file read, and a write only when the marker is absent.

``theme.inject_theme`` also injects a runtime shim; that covers the very first
page-load in a process whose ``index.html`` write lost the race.  A hard refresh
(Ctrl+Shift+R) always lands on the patched file.
"""

from __future__ import annotations

import os

_MARKER = "carnx-ltr-boot"

# Pin navigator.language / navigator.languages to English.  react-aria reads
# `navigator.language` specifically, so that key must resolve to an "en-*" tag.
_SNIPPET = (
    f'<script>/* {_MARKER} */(function(){{try{{'
    "var L='en-US';"
    "Object.defineProperty(navigator,'language',{get:function(){return L;},configurable:true});"
    "Object.defineProperty(navigator,'languages',{get:function(){return [L];},configurable:true});"
    "}}catch(e){}}})();</script>"
)


def _index_html_path() -> str | None:
    try:
        import streamlit
    except Exception:  # pragma: no cover - streamlit always present in this app
        return None
    p = os.path.join(os.path.dirname(streamlit.__file__), "static", "index.html")
    return p if os.path.isfile(p) else None


def apply() -> bool:
    """Patch static/index.html if needed.  Returns True when a write happened."""
    path = _index_html_path()
    if not path:
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
    except OSError:
        return False
    if _MARKER in html:
        return False
    anchor = "<head>"
    i = html.find(anchor)
    if i == -1:
        return False
    patched = html[: i + len(anchor)] + "\n    " + _SNIPPET + html[i + len(anchor) :]
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(patched)
    except OSError:
        return False
    return True


_PATCHED = apply()
