"""
CARN-X  --  install the app identity into the Streamlit shell  (presentation only)
================================================================================
Streamlit gives no hook for the document ``<head>``, so -- exactly like
``ltr_boot`` -- this module rewrites the frontend's static ``index.html`` and
drops the icon set + web-app manifest into the static directory.  The result:

* a real favicon (the engraved ∫ mark, not the 📈 emoji);
* "Add to Home Screen" on iOS / Android installs CARN-X like any native app,
  with the right icon, name, splash colour and standalone chrome;
* ``theme-color`` so the mobile status bar matches the deep-navy plate.

Idempotent (marker-guarded) and self-healing (re-applied every process start, so
a ``pip install -U streamlit`` that ships a fresh ``index.html`` can't undo it).
Nothing here touches the model, the screens, or any widget behaviour.
"""

from __future__ import annotations

import os
import re
import shutil

_MARKER = "carnx-brand-boot"
_APP_NAME = "CARN-X"
_THEME = "#0A0E1C"       # the navy plate -- matches theme.py --ink
_BG = "#05070F"          # splash background -- matches theme.py --ink-edge

# project assets  ->  name inside  static/carnx/
_ICONS = {
    "favicon_32.png": "favicon-32.png",
    "favicon_64.png": "favicon-64.png",
    "icon_180.png": "icon-180.png",
    "icon_192.png": "icon-192.png",
    "icon_512.png": "icon-512.png",
    "icon_maskable_512.png": "icon-maskable-512.png",
}

_MANIFEST = f"""{{
  "name": "{_APP_NAME} \\u2014 probabilistic forecasting",
  "short_name": "{_APP_NAME}",
  "description": "Calibrated probabilistic forecasts for MicroStrategy (MSTR) and Bitcoin.",
  "start_url": "./",
  "scope": "./",
  "display": "standalone",
  "orientation": "any",
  "lang": "en",
  "dir": "ltr",
  "background_color": "{_BG}",
  "theme_color": "{_THEME}",
  "icons": [
    {{ "src": "./carnx/icon-192.png", "sizes": "192x192", "type": "image/png" }},
    {{ "src": "./carnx/icon-512.png", "sizes": "512x512", "type": "image/png" }},
    {{ "src": "./carnx/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }}
  ]
}}
"""

_HEAD = f"""<!-- {_MARKER} -->
    <link rel="icon" type="image/png" sizes="32x32" href="./carnx/favicon-32.png">
    <link rel="icon" type="image/png" sizes="64x64" href="./carnx/favicon-64.png">
    <link rel="icon" type="image/png" sizes="192x192" href="./carnx/icon-192.png">
    <link rel="apple-touch-icon" sizes="180x180" href="./carnx/icon-180.png">
    <link rel="manifest" href="./carnx/manifest.webmanifest">
    <meta name="theme-color" content="{_THEME}">
    <meta name="color-scheme" content="dark">
    <meta name="application-name" content="{_APP_NAME}">
    <meta name="apple-mobile-web-app-title" content="{_APP_NAME}">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
"""


def _static_dir() -> str | None:
    try:
        import streamlit
    except Exception:  # pragma: no cover
        return None
    d = os.path.join(os.path.dirname(streamlit.__file__), "static")
    return d if os.path.isdir(d) else None


def _assets_dir() -> str | None:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    return d if os.path.isdir(d) else None


def _install_assets(static_dir: str) -> bool:
    src = _assets_dir()
    if not src:
        return False
    dst = os.path.join(static_dir, "carnx")
    try:
        os.makedirs(dst, exist_ok=True)
        for a, b in _ICONS.items():
            p = os.path.join(src, a)
            if os.path.isfile(p):
                shutil.copyfile(p, os.path.join(dst, b))
        with open(os.path.join(dst, "manifest.webmanifest"), "w", encoding="utf-8") as fh:
            fh.write(_MANIFEST)
    except Exception:
        return False
    return True


def _patch_index(static_dir: str) -> bool:
    path = os.path.join(static_dir, "index.html")
    try:
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
    except Exception:
        return False
    # drop any earlier carnx-brand-boot block so an updated one always wins
    cleaned = re.sub(
        r"\s*<!-- " + _MARKER + r" -->.*?black-translucent\">\n?",
        "",
        html,
        flags=re.S,
    )
    if _HEAD.strip() in cleaned:
        patched = cleaned
    else:
        anchor = "<head>"
        i = cleaned.find(anchor)
        if i == -1:
            return False
        patched = cleaned[: i + len(anchor)] + "\n    " + _HEAD + cleaned[i + len(anchor):]
    # our favicon links should win over Streamlit's default shortcut icon
    patched = patched.replace('<link rel="shortcut icon" href="./favicon.png" />', "")
    if patched == html:
        return False
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(patched)
    except Exception:
        return False
    return True


def apply() -> bool:
    """Install icons + manifest and patch index.html.  True if anything changed."""
    static_dir = _static_dir()
    if not static_dir:
        return False
    a = _install_assets(static_dir)
    b = _patch_index(static_dir)
    return a or b


try:
    _APPLIED = apply()
except Exception:  # a cosmetic patch must never break app startup
    _APPLIED = False
