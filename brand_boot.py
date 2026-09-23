"""
CARN-X  --  install the app identity into the Streamlit shell  (presentation only)
================================================================================
Streamlit gives no hook for the document ``<head>``. The original approach
here (still attempted first, for a plain local ``streamlit run``) rewrote the
frontend's static ``index.html`` and copied the icon set into the static
directory -- exactly like ``ltr_boot``. That write is silently refused on a
managed host such as Streamlit Community Cloud (confirmed on the user's live
deploy: no favicon, no apple-touch-icon, nothing), so ``render()`` below
installs the same identity a second way that has no filesystem dependency at
all: a small script (delivered through the same base64 + ``eval`` pattern
``scroll_boot`` uses, for the same DOMPurify-strips-any-``<`` reason -- see
that module's docstring) that builds the ``<link>``/``<meta>`` tags with
``document.createElement`` and appends them straight to ``document.head``,
with every icon inlined as a ``data:`` URI so there is nothing left to fetch
or write. Whichever of the two mechanisms a given host allows, the identity
shows up; on a host that allows both, the second is idempotent (window-flag
guarded) and a harmless no-op once the tags already exist.

* a real favicon (the engraved mark, not the default Streamlit icon);
* apple-touch-icon, so "Add to Home Screen" on iOS uses the right icon;
* ``theme-color`` so the mobile status bar matches the deep-navy plate.

Nothing here touches the model, the screens, or any widget behaviour.
"""

from __future__ import annotations

import base64
import os
import re
import shutil

_MARKER = "carnx-brand-boot"
_APP_NAME = "CARN-X"
_THEME = "#0A0A14"       # the ground -- matches theme.py --ink
_BG = "#060610"          # splash background -- matches theme.py --ink-edge

# project assets  ->  name inside  static/carnx/  (file-patch path only)
_ICONS = {
    "favicon_32.png": "favicon-32.png",
    "favicon_64.png": "favicon-64.png",
    "icon_180.png": "icon-180.png",
    "icon_192.png": "icon-192.png",
    "icon_512.png": "icon-512.png",
    "icon_maskable_512.png": "icon-maskable-512.png",
    "hub.png": "hub.png",
    "mark.png": "mark.png",
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
    """Best-effort file patch for a plain local ``streamlit run``.  True if
    anything changed.  Silently does nothing on a host that refuses the
    write (Streamlit Community Cloud) -- ``render()`` is what actually
    covers that case, via the DOM instead of the filesystem."""
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


def _data_uri(filename: str) -> str:
    d = _assets_dir()
    if not d:
        return ""
    path = os.path.join(d, filename)
    try:
        with open(path, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")
    except Exception:
        return ""


def render() -> None:
    """Build the same <link>/<meta> identity tags at runtime, straight in
    document.head, with every icon inlined -- no filesystem write, so it
    works on hosts that refuse ``apply()``'s. Call once per script run,
    anywhere after ``inject_theme()``."""
    try:
        import streamlit as st

        icon32 = _data_uri("favicon_32.png")
        icon180 = _data_uri("icon_180.png")
        if not icon32 and not icon180:
            return  # nothing to inline; leave apply()'s attempt as-is

        js = (
            "(function(){"
            "try{"
            "if(window.__cxBrand) return; window.__cxBrand=true;"
            "var H=document.head;"
            "function mk(tag,attrs){"
            "var e=document.createElement(tag);"
            "for(var k in attrs){ e.setAttribute(k, attrs[k]); }"
            "return e;"
            "}"
            "document.querySelectorAll('link[rel~=icon]').forEach(function(l){ l.remove(); });"
            + (
                f"H.appendChild(mk('link',{{rel:'icon',type:'image/png',sizes:'32x32',href:'{icon32}'}}));"
                if icon32
                else ""
            )
            + (
                f"H.appendChild(mk('link',{{rel:'apple-touch-icon',sizes:'180x180',href:'{icon180}'}}));"
                if icon180
                else ""
            )
            + f"H.appendChild(mk('meta',{{name:'theme-color',content:'{_THEME}'}}));"
            "H.appendChild(mk('meta',{name:'color-scheme',content:'dark'}));"
            f"H.appendChild(mk('meta',{{name:'application-name',content:'{_APP_NAME}'}}));"
            f"H.appendChild(mk('meta',{{name:'apple-mobile-web-app-title',content:'{_APP_NAME}'}}));"
            "H.appendChild(mk('meta',{name:'apple-mobile-web-app-capable',content:'yes'}));"
            "}catch(e){}"
            "})();"
        )
        # same reasoning as scroll_boot.render(): DOMPurify strips a
        # <script> body outright if it contains one literal "<", and a
        # data: URI here would trip that -- so decode-and-eval a base64
        # payload instead of inlining the script text directly.
        b64 = base64.b64encode(js.encode("utf-8")).decode("ascii")
        st.html(f'<script>eval(atob("{b64}"))</script>', unsafe_allow_javascript=True)
    except Exception:  # a cosmetic enhancement must never break the app
        pass
