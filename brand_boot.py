"""
CARN-X  --  install the app identity into the Streamlit shell  (presentation only)
================================================================================
Streamlit gives no hook for the document ``<head>``, so the icon / manifest
links have to be inserted at runtime via a script -- and, separately, they
have to point at a URL Streamlit will actually serve.

Two things were tried and discarded before this:

1. Rewriting Streamlit's own *installed package* ``static/index.html`` and
   copying icon files into its static dir (exactly like ``ltr_boot`` still
   does for the RTL fix). A managed host such as Streamlit Community Cloud
   silently refuses that write -- confirmed on the live deploy: no favicon,
   no apple-touch-icon, nothing.
2. Inlining every icon as a ``data:`` URI in the injected tags, so there was
   nothing left to fetch. That part *did* survive to Cloud -- but iOS
   Safari's "Add to Home Screen" icon fetch does not reliably honour a
   ``data:`` URI for ``apple-touch-icon``; it fell back to a generic icon.

The fix that actually holds on every host: ship the icons as real files in
this repo, under ``static/`` next to this script, and let Streamlit's own
*app-level* static file route (``enableStaticServing = true`` in
``.streamlit/config.toml``) serve them at ``./app/static/<name>`` -- a real,
same-origin, fetchable URL, with zero runtime filesystem write (the files
are just part of the deployed repo, identically on a laptop or on Cloud).
``render()`` still has to build the ``<link>``/``<meta>`` tags with
``document.createElement`` (Streamlit still gives no ``<head>`` hook), via
the same base64 + ``eval`` delivery ``scroll_boot`` uses (DOMPurify strips a
``<script>`` body outright if it contains one literal ``<`` -- see that
module's docstring) -- but the values it writes are now short, ordinary
paths instead of megabyte-scale inlined images.

* a real favicon (the engraved mark, not the default Streamlit icon);
* apple-touch-icon + a web manifest, so "Add to Home Screen" on iOS/Android
  installs CARN-X with the right icon, name and a standalone (chrome-less)
  window instead of opening inside Safari/Chrome;
* ``theme-color`` so the status bar / task-switcher tint matches the app.

Nothing here touches the model, the screens, or any widget behaviour.
"""

from __future__ import annotations

import base64

_APP_NAME = "CARN-X"
_THEME = "#EAF2FF"       # the ground -- matches theme.py --ink-edge
_STATIC = "./app/static"  # Streamlit's app-level static route (see config.toml)


def render() -> None:
    """Build the identity <link>/<meta> tags at runtime, straight in
    document.head, pointing at the real files under static/. Call once per
    script run, anywhere after ``inject_theme()``."""
    try:
        import streamlit as st

        js = (
            "(function(){"
            "try{"
            "if(window.__cxBrand) return; window.__cxBrand=true;"
            "var H=document.head, S='" + _STATIC + "';"
            "function mk(tag,attrs){"
            "var e=document.createElement(tag);"
            "for(var k in attrs){ e.setAttribute(k, attrs[k]); }"
            "return e;"
            "}"
            "document.querySelectorAll('link[rel~=icon]').forEach(function(l){ l.remove(); });"
            "H.appendChild(mk('link',{rel:'icon',type:'image/png',sizes:'32x32',href:S+'/favicon-32.png'}));"
            "H.appendChild(mk('link',{rel:'icon',type:'image/png',sizes:'64x64',href:S+'/favicon-64.png'}));"
            "H.appendChild(mk('link',{rel:'icon',type:'image/png',sizes:'192x192',href:S+'/icon-192.png'}));"
            "H.appendChild(mk('link',{rel:'apple-touch-icon',sizes:'180x180',href:S+'/icon-180.png'}));"
            "H.appendChild(mk('link',{rel:'manifest',href:S+'/manifest.webmanifest'}));"
            f"H.appendChild(mk('meta',{{name:'theme-color',content:'{_THEME}'}}));"
            "H.appendChild(mk('meta',{name:'color-scheme',content:'light'}));"
            f"H.appendChild(mk('meta',{{name:'application-name',content:'{_APP_NAME}'}}));"
            f"H.appendChild(mk('meta',{{name:'apple-mobile-web-app-title',content:'{_APP_NAME}'}}));"
            "H.appendChild(mk('meta',{name:'apple-mobile-web-app-capable',content:'yes'}));"
            "H.appendChild(mk('meta',{name:'mobile-web-app-capable',content:'yes'}));"
            "H.appendChild(mk('meta',{name:'apple-mobile-web-app-status-bar-style',content:'default'}));"
            "}catch(e){}"
            "})();"
        )
        b64 = base64.b64encode(js.encode("utf-8")).decode("ascii")
        st.html(f'<script>eval(atob("{b64}"))</script>', unsafe_allow_javascript=True)
    except Exception:  # a cosmetic enhancement must never break the app
        pass
