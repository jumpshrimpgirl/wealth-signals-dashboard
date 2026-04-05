









Wealth Signals Dashboard - Streamlit UI.

Run: streamlit run app.py
"""

import html
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

from data import fetch_signals

# How long a signal counts as “NEW” in the feed (hours)
NEW_WINDOW_HOURS = 48


def inject_styles() -> None:
    """Global look: soft canvas, typography, cards, metrics, table shell, sidebar."""
    st.markdown(
