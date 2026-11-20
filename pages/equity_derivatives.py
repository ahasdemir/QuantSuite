"""
Equity Derivatives Page — QuantSuite Phase 1
=============================================
Black-Scholes-Merton pricing for dividend-paying equities.
Live BIST spot price fetched via borsapy; other parameters via UI.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis_utils import get_stock_data, get_annualization_factor

from pricing_engine import (
    EquityMarketEnv,
    bsm_price,
    implied_vol,
    option_payoff,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Equity Derivatives | QuantSuite", layout="wide")

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label { color: #94a3b8; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { color: #f1f5f9; font-size: 1.5rem; font-weight: 700; font-family: 'monospace'; }
.metric-sub   { color: #64748b; font-size: 0.72rem; margin-top: 2px; }
.greek-pos    { color: #34d399; }
.greek-neg    { color: #f87171; }
.section-header {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding-bottom: 6px;
    border-bottom: 1px solid #334155;
    margin-bottom: 14px;
}
.live-badge {
    display: inline-block;
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 8px;
    vertical-align: middle;
}
.warn-badge {
    display: inline-block;
    background: #1c1917;
    color: #facc15;
    border: 1px solid #713f12;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Equity Derivatives")
st.caption("Black-Scholes-Merton pricing engine for dividend-paying equities · Greeks & IV solver")

# ─────────────────────────────────────────────────────────────
# Sidebar — inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Equity Option Parameters")

    # Live spot fetch
    st.markdown('<p class="section-header">Underlying Asset</p>', unsafe_allow_html=True)
    ticker_input = st.text_input("BIST Ticker (borsapy)", value="THYAO",
                                 help="Enter a BIST ticker for live spot price via borsapy, or leave blank to enter manually.")
    use_live = st.toggle("Fetch Live Spot", value=True)

    live_spot = None
    live_source = None
    if use_live and ticker_input.strip():
        try:
            import borsapy as bp
            t = bp.Ticker(ticker_input.strip().upper())
            live_spot = float(t.fast_info["last_price"])
            live_source = "borsapy"
        except Exception as e:
            st.warning(f"borsapy fetch failed: {e}")

    if live_spot is not None:
        st.success(f"Live: **{ticker_input.upper()}** = ₺{live_spot:,.2f}")
        default_S = live_spot
    else:
        default_S = 100.0

    S = st.number_input("Spot Price (S) ₺", min_value=0.01, value=float(round(default_S, 2)), step=0.01,
                        format="%.2f",
                        help="Current price of the underlying stock.")

    st.markdown('<p class="section-header">Contract</p>', unsafe_allow_html=True)
    K = st.number_input("Strike Price (K) ₺", min_value=0.01, value=float(round(S * 1.05, 2)), step=0.01, format="%.2f")
    T_days = st.number_input("Days to Maturity", min_value=1, max_value=3650, value=90, step=1)
    T = T_days / 365.0
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)

    st.markdown('<p class="section-header">Market Parameters</p>', unsafe_allow_html=True)
    r_pct = st.slider(
        "Risk-Free Rate r (%)",
        min_value=0.0,
        max_value=60.0,
        value=45.0,
        step=0.25,
        help="Domestic risk-free rate (e.g. TCMB overnight rate). Enter as percentage.",
    )
    q_pct = st.slider(
        "Dividend Yield q (%)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.25,
        help="Continuous dividend yield of the stock.",
    )

    sigma_pct = None
    sigma_note = None
    if ticker_input.strip():
        try:
            ticker_symbol = ticker_input.strip().upper()
            try:
                df_vol = get_stock_data(ticker_symbol, period="1y")
            except Exception:
                df_vol = None

            if df_vol is None or df_vol.empty:
                if not ticker_symbol.endswith(".IS"):
                    df_vol = get_stock_data(f"{ticker_symbol}.IS", period="1y")
                    ticker_symbol = f"{ticker_symbol}.IS"

            if df_vol is None or df_vol.empty:
                raise ValueError("No price history returned")

            annualization_factor = get_annualization_factor(ticker_symbol)
            rolling_vol = df_vol["Log_Return"].rolling(21).std() * np.sqrt(annualization_factor)
            sigma_value = float(rolling_vol.dropna().iloc[-1])
            sigma_pct = sigma_value * 100
            sigma_note = f"{ticker_symbol} · 21d rolling"
        except Exception as e:
            st.warning(f"Volatility fetch failed: {e}")

    if sigma_pct is None:
        sigma_pct = 40.0
        sigma_note = "Default 40% (volatility unavailable)"

    st.metric("Volatility σ (annualized)", f"{sigma_pct:.2f}%")
    if sigma_note:
        st.caption(sigma_note)

    r = r_pct / 100
    q = q_pct / 100
    sigma = sigma_pct / 100

    st.divider()
    st.markdown("### 🔍 IV Solver")
    market_price_input = st.number_input(
        "Market Premium (for IV)", min_value=0.0, value=0.0, step=0.01, format="%.4f",
        help="Enter the observed market option price to back-solve for Implied Volatility. Leave 0 to skip."
    )

# ─────────────────────────────────────────────────────────────
# Pricing
# ─────────────────────────────────────────────────────────────
env = EquityMarketEnv(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
try:
    result = bsm_price(env, option_type)
    price   = result.price
    greeks  = result.greeks
    d1, d2  = result.d1, result.d2
    error_msg = None
except Exception as e:
    error_msg = str(e)
    price = greeks = d1 = d2 = None

# ─────────────────────────────────────────────────────────────
# IV solve (if market price given)
# ─────────────────────────────────────────────────────────────
iv_result = None
iv_error  = None
if market_price_input > 0:
    try:
        iv_result = implied_vol(
            market_price=market_price_input,
            S=S, K=K, T=T, r=r, q=q,
            option_type=option_type,
            model="bsm",
        )
    except Exception as e:
        iv_error = str(e)

# ─────────────────────────────────────────────────────────────
# Layout — header row
# ─────────────────────────────────────────────────────────────
if error_msg:
    st.error(f"Pricing error: {error_msg}")
    st.stop()

col_title, col_badge = st.columns([6, 2])
with col_title:
    st.markdown(f"**{option_type.upper()} option** · Strike ₺{K:,.2f} · {T_days}d to expiry")
with col_badge:
    if live_spot is not None:
        st.markdown(f'<span class="live-badge">● LIVE {ticker_input.upper()}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn-badge">⚠ Manual Spot</span>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# Metric row 1 — BSM Price + Greeks
# ─────────────────────────────────────────────────────────────

def _color(v): return "greek-pos" if v >= 0 else "greek-neg"
def _fmt(v, decimals=4): return f"{v:+.{decimals}f}"

cols = st.columns(6)
metrics = [
    ("BSM Price", f"₺{price:.4f}", "Option Premium"),
    ("Delta Δ",   f"{_fmt(greeks.delta,4)}", f"{'Calls near 0→1' if option_type=='call' else 'Puts near -1→0'}"),
    ("Gamma Γ",   f"{_fmt(greeks.gamma,6)}", "Δ per ₺1 spot move"),
    ("Theta Θ",   f"{_fmt(greeks.theta,4)}/day", "Time decay per day"),
    ("Vega ν",    f"{_fmt(greeks.vega,4)}", "per 1% vol move"),
    ("Rho ρ",     f"{_fmt(greeks.rho,4)}", "per 1% rate move"),
]
for col, (label, value, sub) in zip(cols, metrics):
    color_cls = _color(float(value.replace("₺","").replace("/day","").replace("+","").replace(",","")))
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_cls}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# IV Solver result
# ─────────────────────────────────────────────────────────────
if iv_result is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("📐 Implied Volatility", f"{iv_result*100:.2f}%",
              delta=f"{(iv_result - sigma)*100:+.2f}% vs σ input")
    c2.metric("Input σ", f"{sigma_pct:.2f}%")
    c3.metric("Market Premium", f"₺{market_price_input:.4f}")
elif iv_error:
    st.warning(f"IV Solver: {iv_error}")

# ─────────────────────────────────────────────────────────────
# Charts — Payoff + Greek sensitivity
# ─────────────────────────────────────────────────────────────
st.subheader("📈 Payoff & Sensitivity Analysis")

tab1, tab2, tab3 = st.tabs(["Expiry Payoff", "Delta vs Spot", "Vega vs Volatility"])

spot_range = np.linspace(S * 0.5, S * 1.5, 300)

# Tab 1 — Payoff at expiry
with tab1:
    pnl = option_payoff(spot_range, K, price, option_type, "long")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl, mode="lines", name="P&L at Expiry",
        line=dict(color="#6366f1", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.10)",
    ))
    fig.add_hline(y=0, line_color="#475569", line_dash="dash", line_width=1)
    fig.add_vline(x=S,  line_color="#22d3ee", line_dash="dot",  annotation_text=f"Spot ₺{S:.2f}",  annotation_position="top right")
    fig.add_vline(x=K,  line_color="#f59e0b", line_dash="dot",  annotation_text=f"Strike ₺{K:.2f}", annotation_position="top left")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        title=f"Long {option_type.title()} — Expiry P&L",
        xaxis_title="Spot Price at Expiry (₺)",
        yaxis_title="Net P&L (₺)",
        height=400, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2 — Delta vs Spot
with tab2:
    from pricing_engine import bsm_price as _bsm
    deltas = []
    for s in spot_range:
        try:
            e = EquityMarketEnv(S=s, K=K, T=T, r=r, q=q, sigma=sigma)
            deltas.append(_bsm(e, option_type).greeks.delta)
        except Exception:
            deltas.append(np.nan)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=spot_range, y=deltas, mode="lines", name="Delta",
        line=dict(color="#34d399", width=2.5),
    ))
    fig2.add_vline(x=S, line_color="#22d3ee", line_dash="dot",
                   annotation_text=f"Spot ₺{S:.2f}", annotation_position="top right")
    fig2.add_vline(x=K, line_color="#f59e0b", line_dash="dot",
                   annotation_text=f"Strike ₺{K:.2f}", annotation_position="top left")
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        title="Delta (Δ) vs Spot Price",
        xaxis_title="Spot Price (₺)", yaxis_title="Delta",
        height=400, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

# Tab 3 — Vega vs Volatility
with tab3:
    vol_range = np.linspace(0.05, 2.0, 200)
    vegas = []
    for sv in vol_range:
        try:
            e = EquityMarketEnv(S=S, K=K, T=T, r=r, q=q, sigma=sv)
            vegas.append(_bsm(e, option_type).greeks.vega)
        except Exception:
            vegas.append(np.nan)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=vol_range * 100, y=vegas, mode="lines", name="Vega",
        line=dict(color="#fb923c", width=2.5),
    ))
    fig3.add_vline(x=sigma_pct, line_color="#22d3ee", line_dash="dot",
                   annotation_text=f"σ {sigma_pct:.1f}%", annotation_position="top right")
    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        title="Vega (ν) vs Volatility",
        xaxis_title="Volatility σ (%)", yaxis_title="Vega (per 1% vol move)",
        height=400, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Diagnostics table
# ─────────────────────────────────────────────────────────────
with st.expander("🔬 Model Diagnostics"):
    import pandas as pd
    df_diag = pd.DataFrame({
        "Parameter": ["Spot S", "Strike K", "Maturity T", "Risk-Free r", "Div Yield q", "Volatility σ", "d₁", "d₂", "N(d₁)", "N(d₂)"],
        "Value": [
            f"₺{S:.4f}", f"₺{K:.4f}", f"{T:.6f} yrs ({T_days}d)",
            f"{r_pct:.2f}%", f"{q_pct:.2f}%", f"{sigma_pct:.2f}%",
            f"{d1:.6f}", f"{d2:.6f}",
            f"{float(np.vectorize(lambda x: __import__('scipy.stats',fromlist=['norm']).stats.norm.cdf(x))(d1)):.6f}" if False else f"{float(__import__('scipy.stats', fromlist=['norm']).norm.cdf(d1)):.6f}",
            f"{float(__import__('scipy.stats', fromlist=['norm']).norm.cdf(d2)):.6f}",
        ],
    })
    st.dataframe(df_diag, use_container_width=True, hide_index=True)
