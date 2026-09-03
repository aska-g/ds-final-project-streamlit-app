"""Render a Claude Design (.dc.html) document as a Streamlit page.

st.Page only accepts Python scripts, so each .dc.html gets a thin .py wrapper
that calls render() below. The document loads its runtime with a relative
<script src="../assets/support.js">, which cannot resolve inside the sandboxed
component iframe -- so the script is inlined instead.
"""

from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent
_RUNTIME_TAG = '<script src="../assets/support.js"></script>'


@st.cache_data(show_spinner=False)
def _document(name):
    html = (_ROOT / "pages" / name).read_text()
    runtime = (_ROOT / "assets" / "support.js").read_text()
    return html.replace(_RUNTIME_TAG, f"<script>{runtime}</script>")


_PAGES = [
    ("presentation", "Presentation"),
    ("writeup", "Write Up"),
    ("demo", "Demo"),
    ("video", "Live Video"),
    ("model_compare", "Model Comparison"),
    ("model_performance", "Model Performance"),
]

_FULLSCREEN_CSS = """<style>
.stAppHeader, header[data-testid="stHeader"], [data-testid="stBottom"] { display: none !important; }
.stApp, .stAppViewContainer, .stMain { background: #141414 !important; }
.stAppViewContainer { top: 0 !important; }
.stMainBlockContainer { padding: 0 !important; gap: 0 !important; max-width: 100% !important; }
.stMainBlockContainer > div, .stVerticalBlock { gap: 0 !important; }
iframe.stIFrame {
    position: fixed; inset: 0; border: 0;
    width: 100% !important; height: 100% !important;
}
#dc-nav-zone {
    position: fixed; top: 0; left: 0; right: 0; height: 14px; z-index: 9999;
}
#dc-nav {
    display: flex; gap: 1.25rem; justify-content: center; align-items: center;
    padding: 0.6rem 1rem; background: rgba(20, 20, 20, 0.92);
    font: 500 0.85rem/1 system-ui, sans-serif;
    transform: translateY(-100%); transition: transform 0.2s ease;
}
#dc-nav-zone:hover #dc-nav, #dc-nav:hover { transform: translateY(0); }
#dc-nav a { color: #d8d8d8; text-decoration: none; }
#dc-nav a:hover { color: #fff; }
</style>"""

_NAV = '<div id="dc-nav-zone"><div id="dc-nav">' + "".join(
    f'<a href="{path}" target="_self">{label}</a>' for path, label in _PAGES
) + "</div></div>"


def render(name):
    st.markdown(_FULLSCREEN_CSS + _NAV, unsafe_allow_html=True)
    st.iframe(_document(name), height="stretch")


if __name__ == "__main__":
    doc = _document("HI-VIS Landing v3.dc.html")
    assert _RUNTIME_TAG not in doc, "runtime script tag still relative"
    assert "dc-runtime" in doc, "runtime not inlined"
    print("ok")
