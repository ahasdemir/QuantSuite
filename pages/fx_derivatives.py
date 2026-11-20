"""
FX Derivatives Page — QuantSuite Phase 1
=========================================
Garman-Kohlhagen pricing for FX options.
Live USD/TRY (or any FX) spot price fetched via borsapy.
Domestic (r_d) and foreign (r_f) risk-free rates entered via Streamlit inputs.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pricing_engine import (
    FXMarketEnv,
    gk_price,
    implied_vol,
    option_payoff,
)

# ── Page styling ─────────────────────────────────────────────
st.set_page_config(page_title="FX Derivatives | QuantSuite", layout="wide")

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #3730a3;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label { color: #a5b4fc; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { color: #f1f5f9; font-size: 1.5rem; font-weight: 700; font-family: monospace; }
.metric-sub   { color: #6366f1; font-size: 0.72rem; margin-top: 2px; }
.greek-pos    { color: #34d399; }
.greek-neg    { color: #f87171; }
.section-header {
    color: #e0e7ff;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding-bottom: 6px;
    border-bottom: 1px solid #3730a3;
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
}
.model-info {
    background: #1e1b4b;
    border-left: 4px solid #6366f1;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #c7d2fe;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("💱 FX Derivatives")
st.caption("Garman-Kohlhagen pricing engine for currency options · Domestic & Foreign rate pipeline")

st.markdown("""
<div class="model-info">
<b>Garman-Kohlhagen Model (1983):</b>
The GK model extends Black-Scholes to currency options by treating the foreign risk-free rate r<sub>f</sub>
as a continuous dividend yield on the spot FX rate — the "cost of carry" for holding a foreign currency.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ FX Option Parameters")

    # Live FX spot
    st.markdown('<p class="section-header">FX Pair</p>', unsafe_allow_html=True)

    FX_PAIRS = {
        "USD/TRY": "USD",
        "EUR/TRY": "EUR",
        "GBP/TRY": "GBP",
        "JPY/TRY": "JPY",
        "CHF/TRY": "CHF",
        "CAD/TRY": "CAD",
        "AUD/TRY": "AUD",
        "Gold (gram-altin)": "gram-altin",
        "Brent Crude": "BRENT",
        "Custom": None,
    }

    pair_name = st.selectbox("FX Pair", list(FX_PAIRS.keys()), index=0)
    custom_fx = None
    if pair_name == "Custom":
        custom_fx = st.text_input("borsapy FX symbol", value="USD",
                                  help="Any symbol accepted by borsapy's FX class.")
    fx_symbol = custom_fx if pair_name == "Custom" else FX_PAIRS[pair_name]
    quote_ccy = "TRY" if fx_symbol not in ["BRENT"] else "USD"

    use_live = st.toggle("Fetch Live Spot", value=True)

    live_spot = None
    if use_live and fx_symbol:
        try:
            import borsapy as bp
            fx_obj   = bp.FX(fx_symbol)
            raw      = fx_obj.current
            # borsapy returns either a float/dict depending on symbol
            if isinstance(raw, dict):
                live_spot = float(raw.get("selling") or raw.get("buying") or raw.get("last") or 0)
            else:
                live_spot = float(raw)
        except Exception as e:
            st.warning(f"borsapy fetch failed for {fx_symbol}: {e}")

    if live_spot and live_spot > 0:
        st.success(f"Live: **{pair_name}** = {quote_ccy}{live_spot:,.4f}")
        default_S = live_spot
    else:
        default_S = 35.0

    S = st.number_input(
        f"Spot Rate (S) [{quote_ccy}]",
        min_value=0.0001, value=float(round(default_S, 4)), step=0.001, format="%.4f",
        help="Current spot FX rate (domestic currency per 1 unit of foreign currency)."
    )

    # Contract
    st.markdown('<p class="section-header">Contract</p>', unsafe_allow_html=True)
    K = st.number_input(
        f"Strike Rate (K) [{quote_ccy}]",
        min_value=0.0001, value=float(round(S * 1.05, 4)), step=0.001, format="%.4f",
    )
    T_days = st.number_input("Days to Maturity", min_value=1, max_value=3650, value=90, step=1)
    T = T_days / 365.0
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True,
                           help="Call = right to buy base currency · Put = right to sell base currency")

    # Rate pipeline: standardised r_d / r_f
    st.markdown('<p class="section-header">Interest Rate Pipeline</p>', unsafe_allow_html=True)
    st.caption("Enter annualised continuously-compounded rates.")

    r_d_pct = st.slider(
        "Domestic Rate r_d (%)", 0.0, 60.0, 45.0, 0.25,
        help="Risk-free rate of the quote currency (e.g. TRY = TCMB policy rate ≈ 45%)."
    )
    r_f_pct = st.slider(
        "Foreign Rate r_f (%)", 0.0, 30.0, 4.5, 0.25,
        help="Risk-free rate of the base currency (e.g. USD = Fed Funds ≈ 4.5%, EUR ≈ 3.75%)."
    )
    sigma_pct = st.slider(
        "Volatility σ (%)", 1.0, 150.0, 25.0, 0.5,
        help="Annualised FX volatility (USD/TRY 1-year realized vol ≈ 20-40%)."
    )

    r_d    = r_d_pct / 100
    r_f    = r_f_pct / 100
    sigma  = sigma_pct / 100

    st.divider()
    st.markdown("### 🔍 IV Solver")
    market_price_input = st.number_input(
        "Market Premium (for IV)", min_value=0.0, value=0.0, step=0.0001, format="%.4f",
        help="Observed market option price to back-solve for Implied Volatility."
    )

# ─────────────────────────────────────────────────────────────
# Pricing
# ─────────────────────────────────────────────────────────────
env = FXMarketEnv(S=S, K=K, T=T, r_d=r_d, r_f=r_f, sigma=sigma)
try:
    result   = gk_price(env, option_type)
    price    = result.price
    greeks   = result.greeks
    d1, d2   = result.d1, result.d2
    error_msg = None
except Exception as e:
    error_msg = str(e)
    price = greeks = d1 = d2 = None

# IV solve
iv_result = None
iv_error  = None
if market_price_input > 0:
    try:
        iv_result = implied_vol(
            market_price=market_price_input,
            S=S, K=K, T=T,
            r=r_d, q=r_f,
            option_type=option_type,
            model="gk",
        )
    except Exception as e:
        iv_error = str(e)

# ─────────────────────────────────────────────────────────────
# Header row
# ─────────────────────────────────────────────────────────────
if error_msg:
    st.error(f"Pricing error: {error_msg}")
    st.stop()

col_title, col_badge = st.columns([6, 2])
with col_title:
    st.markdown(f"**{pair_name} {option_type.upper()} option** · Strike {quote_ccy}{K:,.4f} · {T_days}d to expiry")
with col_badge:
    if live_spot and live_spot > 0:
        st.markdown(f'<span class="live-badge">● LIVE {pair_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn-badge">⚠ Manual Spot</span>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# Greeks row
# ─────────────────────────────────────────────────────────────
def _color(v):
    try: return "greek-pos" if float(str(v).replace("+","").replace(",","").split("/")[0].replace("₺","").replace("$","")) >= 0 else "greek-neg"
    except: return ""

def _fmt(v, decimals=4): return f"{v:+.{decimals}f}"

currency_sym = "₺" if quote_ccy == "TRY" else "$"

cols = st.columns(6)
metrics = [
    ("GK Price", f"{currency_sym}{price:.4f}", "Option Premium"),
    ("Delta Δ",  f"{_fmt(greeks.delta,4)}", "hedge ratio"),
    ("Gamma Γ",  f"{_fmt(greeks.gamma,6)}", "Δ per 1-unit spot move"),
    ("Theta Θ",  f"{_fmt(greeks.theta,4)}/day", "time decay"),
    ("Vega ν",   f"{_fmt(greeks.vega,4)}", "per 1% vol move"),
    ("Rho ρ",    f"{_fmt(greeks.rho,4)}", "per 1% r_d move"),
]
for col, (label, value, sub) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Rate pipeline display
c1, c2, c3 = st.columns(3)
c1.metric("Domestic Rate r_d", f"{r_d_pct:.2f}%", help="Quote currency (TRY) rate")
c2.metric("Foreign Rate r_f",  f"{r_f_pct:.2f}%", help="Base currency (USD/EUR etc.) rate")
c3.metric("Cost of Carry", f"{(r_d_pct - r_f_pct):.2f}%",
          delta=f"r_d - r_f",
          help="Interest rate differential. Positive = quote appreciates vs foreign (carry trade)")

# IV solver
if iv_result is not None:
    st.success(f"📐 **Implied Volatility = {iv_result*100:.3f}%** (NR solver converged) · Input σ = {sigma_pct:.2f}%")
elif iv_error:
    st.warning(f"IV Solver: {iv_error}")

st.divider()

# ─────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────
st.subheader("📉 Payoff & Sensitivity Analysis")

spot_range = np.linspace(S * 0.5, S * 1.5, 300)

tab1, tab2, tab3, tab4 = st.tabs(["Expiry Payoff", "Delta vs Spot", "Vega vs Vol", "Rate Surface"])

# Tab 1 — Payoff
with tab1:
    pnl = option_payoff(spot_range, K, price, option_type, "long")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl, mode="lines", name="P&L at Expiry",
        line=dict(color="#818cf8", width=2.5),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.10)",
    ))
    fig.add_hline(y=0, line_color="#475569", line_dash="dash", line_width=1)
    fig.add_vline(x=S, line_color="#22d3ee", line_dash="dot",
                  annotation_text=f"Spot {S:.4f}", annotation_position="top right")
    fig.add_vline(x=K, line_color="#f59e0b", line_dash="dot",
                  annotation_text=f"Strike {K:.4f}", annotation_position="top left")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        title=f"Long {option_type.title()} — Expiry P&L ({pair_name})",
        xaxis_title=f"FX Rate at Expiry ({quote_ccy})",
        yaxis_title=f"Net P&L ({quote_ccy})",
        height=420, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2 — Delta vs Spot
with tab2:
    from pricing_engine import gk_price as _gk
    deltas = []
    for s in spot_range:
        try:
            e = FXMarketEnv(S=s, K=K, T=T, r_d=r_d, r_f=r_f, sigma=sigma)
            deltas.append(_gk(e, option_type).greeks.delta)
        except Exception:
            deltas.append(np.nan)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=spot_range, y=deltas, mode="lines", name="Delta",
        line=dict(color="#34d399", width=2.5),
    ))
    fig2.add_vline(x=S, line_color="#22d3ee", line_dash="dot",
                   annotation_text=f"Spot {S:.4f}", annotation_position="top right")
    fig2.add_vline(x=K, line_color="#f59e0b", line_dash="dot",
                   annotation_text=f"Strike {K:.4f}", annotation_position="top left")
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        title="Delta (Δ) vs FX Spot Rate",
        xaxis_title=f"FX Spot Rate ({quote_ccy})",
        yaxis_title="Delta",
        height=420, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

# Tab 3 — Vega vs Vol
with tab3:
    vol_range = np.linspace(0.05, 1.5, 200)
    vegas = []
    for sv in vol_range:
        try:
            e = FXMarketEnv(S=S, K=K, T=T, r_d=r_d, r_f=r_f, sigma=sv)
            vegas.append(_gk(e, option_type).greeks.vega)
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
        xaxis_title="Volatility σ (%)", yaxis_title="Vega",
        height=420, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

# Tab 4 — Price surface over r_d vs r_f
with tab4:
    st.caption("Option price sensitivity to the domestic vs foreign rate differential.")
    rd_range = np.linspace(0, 0.60, 30)
    rf_range = np.linspace(0, 0.20, 30)
    Z = np.zeros((len(rd_range), len(rf_range)))
    for i, rd in enumerate(rd_range):
        for j, rf in enumerate(rf_range):
            try:
                e = FXMarketEnv(S=S, K=K, T=T, r_d=rd, r_f=rf, sigma=sigma)
                Z[i, j] = _gk(e, option_type).price
            except Exception:
                Z[i, j] = np.nan
    fig4 = go.Figure(data=go.Surface(
        x=rf_range * 100, y=rd_range * 100, z=Z,
        colorscale="Plasma",
        colorbar=dict(title=f"Price ({quote_ccy})"),
    ))
    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        title=f"GK Price Surface — r_d vs r_f ({option_type.title()})",
        scene=dict(
            xaxis_title="Foreign Rate r_f (%)",
            yaxis_title="Domestic Rate r_d (%)",
            zaxis_title=f"Option Price ({quote_ccy})",
            bgcolor="#0f172a",
        ),
        height=520, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────
with st.expander("🔬 Model Diagnostics (Garman-Kohlhagen)"):
    from scipy.stats import norm
    df_diag = pd.DataFrame({
        "Parameter": ["Spot S", "Strike K", "Maturity T", "Dom. Rate r_d", "For. Rate r_f", "Carry (r_d - r_f)", "Volatility σ", "d₁", "d₂", "N(d₁)", "N(d₂)"],
        "Value": [
            f"{currency_sym}{S:.4f}", f"{currency_sym}{K:.4f}", f"{T:.6f} yrs ({T_days}d)",
            f"{r_d_pct:.2f}%", f"{r_f_pct:.2f}%", f"{r_d_pct - r_f_pct:.2f}%",
            f"{sigma_pct:.2f}%",
            f"{d1:.6f}", f"{d2:.6f}",
            f"{norm.cdf(d1):.6f}", f"{norm.cdf(d2):.6f}",
        ],
    })
    st.dataframe(df_diag, use_container_width=True, hide_index=True)
