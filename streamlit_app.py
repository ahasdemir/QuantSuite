import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from analysis_utils import (
    get_stock_data,
    add_indicators,
    volatility_analysis,
    get_portfolio_history,
    portfolio_performance_with_data,
    calculate_parametric_var,
    calculate_historical_var,
    parametric_var_portfolio,
    historical_var_portfolio,
    geometric_brownian_motion,
    efficient_frontier_analysis_with_monte_carlo,
)

st.set_page_config(
    page_title="The Algo Risk Monitor",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

home = st.Page("pages/home.py", title="Home", icon=":material/home:")
equity_derivatives = st.Page(
    "pages/equity_derivatives.py",
    title="Equity Derivatives",
    icon=":material/analytics:",
)
fx_derivatives = st.Page(
    "pages/fx_derivatives.py",
    title="FX Derivatives",
    icon=":material/currency_exchange:",
)
volatility_analysis = st.Page(
    "pages/volatility_analysis.py",
    title="Volatility Analysis",
    icon=":material/show_chart:",
)
strategy_builder = st.Page(
    "pages/strategy_builder.py",
    title="Strategy Builder",
    icon=":material/construction:",
)

pg = st.navigation(
    [
        home,
        equity_derivatives,
        fx_derivatives,
        volatility_analysis,
        strategy_builder,
    ]
)
pg.run()

# Footer
FOOTER_GITHUB_URL = "https://github.com/ahasdemir"
FOOTER_LINKEDIN_URL = "https://www.linkedin.com/in/ahasdemir/"

st.markdown(
    f"""
    <style>
    .arm-footer {{
        text-align: center;
        padding: 12px 0 18px 0;
        color: #6b7280;
        font-size: 0.95rem;
    }}
    .arm-footer a {{
        color: #2563eb;
        text-decoration: none;
        margin: 0 8px;
        font-weight: 600;
    }}
    .arm-footer a:hover {{ text-decoration: underline; }}
    </style>
    <div class="arm-footer">
        MIT License 2025 Ahmet Hasdemir —
        <a href="{FOOTER_GITHUB_URL}" target="_blank">GitHub</a>
        <a href="{FOOTER_LINKEDIN_URL}" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)