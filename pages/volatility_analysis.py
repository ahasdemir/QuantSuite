"""
Volatility Analysis Page — QuantSuite
======================================
Realized volatility → parametric implied-vol surface, smile cross-sections,
and second-order BSM Greeks (Gamma / Vanna) heatmaps.

- Base vol: realized annualized vol from daily log returns (yfinance via analysis_utils)
- IV(K, T) = sigma_base * (1 + skew * m + smile * m^2),  m = ln(K/S)/sqrt(T)
  → clearly-labelled PARAMETRIC surface (no live option chain exists)
- Gamma & Vanna computed analytically from closed-form BSM formulas
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm

from pricing_engine import bsm_price_scalar, EquityMarketEnv, bsm_price, implied_vol
from analysis_utils import get_stock_data, get_annualization_factor

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Volatility Analysis | QuantSuite", layout="wide")

# Dark dashboard styling (kept in sync with pages/equity_derivatives.py)
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

st.title("🌡️ Volatility Analysis")
st.caption("Realized vol → parametric IV surface · smile & skew · second-order Greeks (Γ, Vanna)")

# ─────────────────────────────────────────────────────────────
# Constants & model helpers
# ─────────────────────────────────────────────────────────────
DEFAULT_TICKER = "AAPL"
FALLBACK_SIGMA = 0.25
FALLBACK_SPOT = 100.0
ATM_TENOR = 3 / 12          # 3 months, for the ATM summary strip
SMILE_TENORS = [("1M", 1 / 12), ("3M", 3 / 12), ("6M", 6 / 12)]
SMILE_COLORS = {"1M": "#22d3ee", "3M": "#6366f1", "6M": "#fb923c"}


def parametric_iv(K, S, T, sigma_base, skew, smile):
    """
    Parametric implied-vol smile/skew model (NOT market quotes):

        IV(K, T) = sigma_base * (1 + skew * m + smile * m^2),
        m = log(K / S) / sqrt(T)   → standardised log-moneyness

    Negative `skew` lifts the put wing (classic equity skew); `smile` adds
    symmetric convexity. Wings steepen as T → 0 via the 1/sqrt(T) scaling.
    Vectorised over K; output clipped to a sane [1%, 300%] band.
    """
    m = np.log(np.asarray(K, dtype=float) / float(S)) / np.sqrt(float(T))
    iv = float(sigma_base) * (1.0 + float(skew) * m + float(smile) * m ** 2)
    return np.clip(iv, 0.01, 3.0)


def bsm_d1_d2(S, K, T, r, q, sigma):
    """d1 / d2 of the Black-Scholes-Merton model (matches pricing_engine._d1_d2)."""
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def bsm_gamma(S, K, T, r, q, sigma):
    """Second-order Greek: Gamma = exp(-qT) · φ(d1) / (S · σ · √T)."""
    d1, _ = bsm_d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_vanna(S, K, T, r, q, sigma):
    """Second-order Greek: Vanna = ∂ν/∂S = ∂Δ/∂σ = -exp(-qT) · φ(d1) · d2 / σ."""
    d1, d2 = bsm_d1_d2(S, K, T, r, q, sigma)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


# ─────────────────────────────────────────────────────────────
# Data fetch (never raises — returns a context dict or an error)
# ─────────────────────────────────────────────────────────────
def fetch_market_context(symbol):
    """
    Fetch 1y of price history and build the volatility context.
    Returns (ctx, None) on success, (None, error_message) on failure.
    """
    try:
        df = get_stock_data(symbol, period="1y")
        if df is None or df.empty:
            raise ValueError("empty price history returned")

        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # safety: flatten stray MultiIndex
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            raise ValueError("no valid close prices")

        log_ret = df["Log_Return"].dropna()
        if len(log_ret) < 30:
            raise ValueError(f"not enough history ({len(log_ret)} returns)")

        ann = get_annualization_factor(symbol)   # 252 for stocks, 365 for crypto
        realized = float(log_ret.std() * np.sqrt(ann))
        if not np.isfinite(realized) or realized <= 0:
            raise ValueError("degenerate volatility estimate")
        realized = float(np.clip(realized, 0.01, 3.0))

        spot = float(close.iloc[-1])
        if not np.isfinite(spot) or spot <= 0:
            raise ValueError("invalid last close")

        return {
            "symbol": symbol,
            "spot": spot,
            "sigma_base": realized,
            "ann": int(ann),
            "n_obs": int(len(log_ret)),
            "last_date": str(pd.Timestamp(df.index[-1]).date()),
            "source": "live",
            "note": "yfinance · 1y daily log returns",
        }, None
    except Exception as e:  # offline / bad ticker / API hiccup — anything
        return None, f"{type(e).__name__}: {e}"


def fallback_context(symbol, error):
    """Context used when no market data is available, so the page never crashes."""
    return {
        "symbol": symbol,
        "spot": FALLBACK_SPOT,
        "sigma_base": FALLBACK_SIGMA,
        "ann": 252,
        "n_obs": 0,
        "last_date": "—",
        "source": "fallback",
        "note": error,
    }


def resolve_symbol(symbol):
    """Try the symbol as typed; fall back to the .IS (BIST) suffix like the equity page."""
    ctx, err = fetch_market_context(symbol)
    if ctx is None and symbol and not symbol.endswith(".IS"):
        ctx_is, err_is = fetch_market_context(f"{symbol}.IS")
        if ctx_is is not None:
            return ctx_is, None
        err = err_is
    return ctx, err


# ─────────────────────────────────────────────────────────────
# Sidebar — inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌡️ Volatility Surface")

    st.markdown('<p class="section-header">Underlying</p>', unsafe_allow_html=True)
    ticker_input = st.text_input(
        "Ticker (yfinance)", value=DEFAULT_TICKER,
        help="Any yfinance symbol — e.g. AAPL, MSFT, THYAO, BTC-USD.",
    )
    fetch_pressed = st.button(
        "📥 Fetch Data & Rebuild Surface", type="primary", use_container_width=True,
    )

    st.markdown('<p class="section-header">Market Parameters</p>', unsafe_allow_html=True)
    r_pct = st.slider("Risk-Free Rate r (%)", 0.0, 30.0, 4.5, 0.25,
                      help="Annualised continuously-compounded risk-free rate.")
    q_pct = st.slider("Dividend Yield q (%)", 0.0, 15.0, 0.5, 0.25,
                     help="Continuous dividend yield of the underlying.")

    st.markdown('<p class="section-header">Parametric Skew / Smile</p>', unsafe_allow_html=True)
    skew = st.slider(
        "Skew Strength", -0.60, 0.60, -0.15, 0.05, format="%.2f",
        help="Coefficient on m = ln(K/S)/√T. Negative → put wing trades rich (equity skew).",
    )
    smile = st.slider(
        "Smile Curvature", 0.00, 0.60, 0.20, 0.05, format="%.2f",
        help="Quadratic (m²) term — adds symmetric convexity on top of the skew.",
    )
    strike_range_pct = st.slider("Strike Range (± % of spot)", 10, 40, 25, 5,
                                help="Width of the strike axis on the surface & smile charts.")
    horizon_months = st.slider("Surface Horizon (months)", 3, 24, 12, 1,
                               help="Longest maturity shown on the surface.")

r = r_pct / 100.0
q = q_pct / 100.0
symbol_input = ticker_input.strip().upper() or DEFAULT_TICKER

# ─────────────────────────────────────────────────────────────
# Data context — button-driven, with a first-load auto attempt
# ─────────────────────────────────────────────────────────────
if "vol_ctx" not in st.session_state:
    st.session_state["vol_ctx"] = None
if "vol_fetch_error" not in st.session_state:
    st.session_state["vol_fetch_error"] = None

if fetch_pressed:
    ctx_new, err = resolve_symbol(symbol_input)
    if ctx_new is not None:
        st.session_state["vol_ctx"] = ctx_new
        st.session_state["vol_fetch_error"] = None
    else:
        if st.session_state["vol_ctx"] is None:
            st.session_state["vol_ctx"] = fallback_context(symbol_input, err)
        st.session_state["vol_fetch_error"] = err

if st.session_state["vol_ctx"] is None:  # first load — try the default once
    ctx_new, err = resolve_symbol(symbol_input)
    if ctx_new is not None:
        st.session_state["vol_ctx"] = ctx_new
    else:
        st.session_state["vol_ctx"] = fallback_context(symbol_input, err)
        st.session_state["vol_fetch_error"] = err

if fetch_pressed and st.session_state["vol_fetch_error"]:
    st.warning(
        f"Data fetch failed for **{symbol_input}** — {st.session_state['vol_fetch_error']} "
        + ("Keeping previously loaded data." if st.session_state["vol_ctx"]["source"] == "live" else "")
    )

ctx = st.session_state["vol_ctx"]
spot = float(ctx["spot"])
sigma_base = float(ctx["sigma_base"])

if ctx["source"] == "fallback":
    st.warning(
        f"⚠️ No market data for **{ctx['symbol']}** ({ctx['note']}). "
        f"Falling back to base σ = {FALLBACK_SIGMA:.0%} and spot = {FALLBACK_SPOT:,.0f} — "
        "everything below is a parametric illustration, not market data."
    )

# ─────────────────────────────────────────────────────────────
# Header row + summary metric cards
# ─────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([6, 2])
with col_title:
    if ctx["source"] == "live":
        st.markdown(
            f"**{ctx['symbol']}** · spot **{spot:,.2f}** · base σ **{sigma_base:.2%}** · "
            f"{ctx['n_obs']} daily log returns · √{ctx['ann']} annualisation"
        )
    else:
        st.markdown(f"**{ctx['symbol']}** · fallback spot **{spot:,.2f}** · base σ **{sigma_base:.2%}**")
with col_badge:
    if ctx["source"] == "live":
        st.markdown(f'<span class="live-badge">● LIVE {ctx["symbol"]}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn-badge">⚠ FALLBACK σ=25%</span>', unsafe_allow_html=True)

st.divider()

# ATM reference option — uses the project's pricing engine directly
env_atm = EquityMarketEnv(S=spot, K=spot, T=ATM_TENOR, r=r, q=q, sigma=sigma_base)
res_atm = bsm_price(env_atm, "call")
gamma_atm = float(bsm_gamma(spot, spot, ATM_TENOR, r, q, sigma_base))
vanna_atm = float(bsm_vanna(spot, spot, ATM_TENOR, r, q, sigma_base))
iv_atm = float(parametric_iv(spot, spot, ATM_TENOR, sigma_base, skew, smile))

cards = [
    ("Spot", f"{spot:,.2f}", f"last close · {ctx['last_date']}"),
    ("Realized σ (base)", f"{sigma_base:.2%}", f"ann. log-return std · √{ctx['ann']}"),
    ("ATM 3M IV", f"{iv_atm:.2%}", "parametric · equals base σ"),
    ("ATM 3M Call (BSM)", f"{float(res_atm.price):,.2f}", f"r={r_pct:.2f}% · q={q_pct:.2f}%"),
    ("Gamma (ATM 3M)", f"{gamma_atm:.4f}", "per 1.0 of spot"),
    ("Vanna (ATM 3M)", f"{vanna_atm:+.4f}", "∂Δ/∂σ = ∂ν/∂S"),
]
cols = st.columns(6)
for col, (label, value, sub) in zip(cols, cards):
    sign_cls = ""
    if label.startswith("Vanna"):
        sign_cls = "greek-pos" if vanna_atm >= 0 else "greek-neg"
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {sign_cls}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Section 1 — Parametric 3D volatility surface
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">🗺️ Parametric 3D Volatility Surface</p>', unsafe_allow_html=True)

k_lo = spot * (1 - strike_range_pct / 100)
k_hi = spot * (1 + strike_range_pct / 100)
horizon_yrs = horizon_months / 12.0

strikes = np.linspace(k_lo, k_hi, 40)
maturities = np.linspace(1 / 12, horizon_yrs, 22)
iv_grid = np.array([parametric_iv(strikes, spot, T, sigma_base, skew, smile)
                    for T in maturities])          # shape: (len(maturities), len(strikes))

fig_surf = go.Figure(data=[go.Surface(
    x=strikes, y=maturities, z=iv_grid,
    colorscale="Viridis",
    colorbar=dict(title="Implied Vol", tickformat=".0%", len=0.8),
    hovertemplate="Strike: %{x:,.2f}<br>Maturity: %{y:.2f} yrs<br>IV: %{z:.2%}<extra></extra>",
    contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
)])
fig_surf.update_layout(
    title=f"Implied Volatility Surface — {ctx['symbol']} (parametric, base σ = {sigma_base:.1%})",
    template="plotly_dark",
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    scene=dict(
        xaxis=dict(title="Strike (K)", backgroundcolor="#0f172a", gridcolor="#334155"),
        yaxis=dict(title="Maturity (years)", backgroundcolor="#0f172a", gridcolor="#334155"),
        zaxis=dict(title="Implied Volatility", backgroundcolor="#0f172a",
                   gridcolor="#334155", tickformat=".0%"),
    ),
    scene_camera=dict(eye=dict(x=1.7, y=-1.4, z=0.9)),
    height=520, margin=dict(t=60, b=40, l=20, r=20),
)
st.plotly_chart(fig_surf, use_container_width=True)
st.caption(
    f"⚠️ **Parametric model surface — not market quotes.** No live option chain is used: "
    f"IV(K,T) = σ_base·(1 + skew·m + smile·m²) with m = ln(K/S)/√T, calibrated only to the "
    f"realized base σ = {sigma_base:.2%} (skew = {skew:+.2f}, smile = {smile:.2f}). "
    "The surface is smooth and plausible, but it is a model, not traded prices."
)

# ─────────────────────────────────────────────────────────────
# Section 2 — Volatility smile / skew cross-sections
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📈 Volatility Smile / Skew — IV vs Strike</p>', unsafe_allow_html=True)

strikes_fine = np.linspace(k_lo, k_hi, 90)
fig_smile = go.Figure()
for label, T in SMILE_TENORS:
    iv_line = parametric_iv(strikes_fine, spot, T, sigma_base, skew, smile)
    fig_smile.add_trace(go.Scatter(
        x=strikes_fine, y=iv_line, mode="lines", name=label,
        line=dict(color=SMILE_COLORS[label], width=2.5),
        hovertemplate=f"Strike %{{x:,.2f}}<br>IV %{{y:.2%}}<extra>{label}</extra>",
    ))
fig_smile.add_vline(x=spot, line_color="#94a3b8", line_dash="dot",
                    annotation_text="ATM", annotation_position="top left")
fig_smile.update_layout(
    title=f"Smile Cross-Sections — {ctx['symbol']} (parametric, 1M / 3M / 6M)",
    xaxis_title="Strike (K)",
    yaxis=dict(title="Implied Volatility", tickformat=".0%"),
    template="plotly_dark",
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    hovermode="x unified",
    height=450, margin=dict(t=60, b=40),
)
st.plotly_chart(fig_smile, use_container_width=True)
st.caption(
    "Short maturities show the steepest skew because m = ln(K/S)/√T blows up as T → 0; "
    "the smile flattens with maturity — the classic equity surface shape."
)

# ─────────────────────────────────────────────────────────────
# Section 3 — Second-order Greeks heatmaps (Gamma & Vanna)
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">🔬 Second-Order Greeks — Gamma & Vanna (BSM, analytic)</p>',
            unsafe_allow_html=True)

spot_grid = np.linspace(k_lo, k_hi, 25)
mat_grid = np.linspace(1 / 12, max(horizon_yrs, 0.75), 12)
gamma_grid = np.zeros((len(mat_grid), len(spot_grid)))
vanna_grid = np.zeros_like(gamma_grid)
for i, T in enumerate(mat_grid):
    for j, s in enumerate(spot_grid):
        gamma_grid[i, j] = bsm_gamma(s, spot, T, r, q, sigma_base)   # K fixed at ATM
        vanna_grid[i, j] = bsm_vanna(s, spot, T, r, q, sigma_base)


def _greek_heatmap(z, title, zlabel, fmt):
    """Spot × Maturity heatmap with a diverging RdBu_r scale centred on zero."""
    return go.Figure(go.Heatmap(
        x=spot_grid, y=mat_grid, z=z,
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title=zlabel),
        hovertemplate=f"Spot %{{x:,.2f}}<br>Maturity %{{y:.2f}} yrs<br>{zlabel} %{{z:{fmt}}}<extra></extra>",
    ))


hcol1, hcol2 = st.columns(2)
with hcol1:
    fig_gamma = _greek_heatmap(gamma_grid, "Gamma Γ — Spot × Maturity", "Γ", ".6f")
    fig_gamma.update_layout(
        title="Gamma Γ — Spot × Maturity (K = ATM)",
        xaxis_title="Spot", yaxis=dict(title="Maturity (years)", tickformat=".2f"),
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=450, margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_gamma, use_container_width=True)
with hcol2:
    fig_vanna = _greek_heatmap(vanna_grid, "Vanna — Spot × Maturity", "Vanna", ".4f")
    fig_vanna.update_layout(
        title="Vanna (∂Δ/∂σ = ∂ν/∂S) — Spot × Maturity (K = ATM)",
        xaxis_title="Spot", yaxis=dict(title="Maturity (years)", tickformat=".2f"),
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=450, margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_vanna, use_container_width=True)

st.caption(
    "Closed-form BSM second-order Greeks over Spot × Maturity with K fixed at the ATM strike "
    f"({spot:,.2f}) and σ = the realized base vol ({sigma_base:.2%}). "
    "Diverging scale: red = positive, blue = negative. Gamma peaks at short maturity near the "
    "strike; Vanna flips sign where d₂ = 0 (≈ forward-ATM)."
)

# ─────────────────────────────────────────────────────────────
# Engine round-trip check — parametric IV ↔ bsm_price_scalar ↔ implied_vol
# ─────────────────────────────────────────────────────────────
with st.expander("🔬 Engine Round-Trip Check — parametric IV → BSM price → implied_vol"):
    st.caption(
        "Prices calls with `bsm_price_scalar` at the parametric IV, then recovers the IV with "
        "the Newton-Raphson solver (`implied_vol`). The recovered IV should match the parametric "
        "IV exactly — proving the surface is consistent with the project's pricing engine."
    )
    rows = []
    for k_mult, T_chk in [(0.90, 1 / 6), (1.00, 1 / 4), (1.05, 1 / 2), (1.10, 1.0)]:
        try:
            K_chk = spot * k_mult
            iv_chk = float(parametric_iv(K_chk, spot, T_chk, sigma_base, skew, smile))
            px_chk = bsm_price_scalar(spot, K_chk, T_chk, r, q, iv_chk, "call")
            iv_back = implied_vol(
                market_price=px_chk, S=spot, K=K_chk, T=T_chk, r=r, q=q,
                option_type="call", model="bsm", initial_guess=iv_chk,
            )
            rows.append({
                "Strike": f"{K_chk:,.2f}",
                "Moneyness": f"{k_mult:.0%}",
                "Maturity (yrs)": f"{T_chk:.2f}",
                "Parametric IV": f"{iv_chk:.4%}",
                "BSM Premium": f"{px_chk:,.4f}",
                "Recovered IV": f"{iv_back:.4%}",
                "Abs. Error": f"{abs(iv_back - iv_chk):.2e}",
            })
        except Exception as e:
            rows.append({"Strike": f"{spot * k_mult:,.2f}", "Moneyness": f"{k_mult:.0%}",
                         "Maturity (yrs)": f"{T_chk:.2f}", "Error": str(e)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "QuantSuite · Volatility Analysis — parametric surface calibrated to realized vol. "
    "To use market-quoted surfaces, feed observed option premiums into `implied_vol`."
)
