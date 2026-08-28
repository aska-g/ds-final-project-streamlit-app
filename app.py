"""P0 Safety — Streamlit app entry point (page router)."""

import streamlit as st

st.set_page_config(page_title="HI-VIS — PPE Compliance", layout="wide")

presentation_page = st.Page("pages/HI-VIS Landing v3.dc.html", title="HI-VIS Presentation", default=True)
writeup_page=st.Page("pages/HI-VIS Landing.dc.html", title="Write Up")
demo_page = st.Page("pages/demo.py", title="Demo")
compare_page = st.Page("pages/model_compare.py", title="Model Comparison")
performance_page = st.Page("pages/model_performance.py", title="Model Performance")

st.navigation([demo_page, compare_page, performance_page], position="top").run()
