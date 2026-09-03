"""P0 Safety — Streamlit app entry point (page router)."""

import streamlit as st

st.set_page_config(page_title="SuperVisor — PPE Compliance", layout="wide")

presentation_page = st.Page("pages/presentation.py", title="SuperVisor Presentation", default=True)
writeup_page = st.Page("pages/writeup.py", title="Write Up")
demo_page = st.Page("pages/demo.py", title="Demo")
video_page = st.Page("pages/video.py", title="Live Video")
compare_page = st.Page("pages/model_compare.py", title="Model Comparison")
historical_page = st.Page("pages/historical.py", title="Historical Data")
performance_page = st.Page("pages/model_performance.py", title="Model Performance")

st.navigation([presentation_page, writeup_page, demo_page, video_page, compare_page, historical_page, performance_page], position="top").run()
