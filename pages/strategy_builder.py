"""
Strategy Builder Page — QuantSuite
==================================
Multi-leg option strategy builder: preset structures (straddle, strangle,
spreads, iron condor, butterfly), BSM-priced legs, payoff at expiry and
P&L risk metrics (net premium, max profit/loss, breakevens).
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pricing_engine import bsm_price_scalar, option_payoff

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Strategy Builder | QuantSuite", layout="wide")

# Dark-theme CSS (matched to pages/equity_derivatives.py)
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

st.title("🎯 Strategy Builder")
st.caption("Multi-leg option strategies · BSM-priced legs · payoff at expiry & risk metrics")

# ─────────────────────────────────────────────────────────────
# Sidebar — market parameters
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Market Parameters")

    st.markdown('<p class="section-header">Underlying</p>', unsafe_allow_html=True)
    S = st.number_input("Spot Price (S) ₺", min_value=0.01, value=100.0,
                        step=0.01, format="%.2f",
                        help="Current spot price of the underlying. The payoff window spans 0.5S → 1.5S.")

    st.markdown('<p class="section-header">Rates & Volatility</p>', unsafe_allow_html=True)
    r_pct = st.slider("Risk-Free Rate r (%)", min_value=0.0, max_value=60.0,
                      value=5.0, step=0.25,
                      help="Annualised, continuously compounded domestic rate.")
    q_pct = st.slider("Dividend Yield q (%)", min_value=0.0, max_value=20.0,
                      value=0.0, step=0.25,
                      help="Continuous dividend yield of the underlying.")
    sigma_pct = st.slider("Implied Volatility σ (%)", min_value=1.0, max_value=200.0,
                          value=25.0, step=0.5,
                          help="Annualised volatility used to BSM-price the legs.")

    st.markdown('<p class="section-header">Time</p>', unsafe_allow_html=True)
    T = st.number_input("Time to Expiry T (years)", min_value=0.01, max_value=10.0,
                        value=0.25, step=0.01, format="%.2f",
                        help="Time to expiry in years (e.g. 0.25 = 3 months).")

r = r_pct / 100.0
q = q_pct / 100.0
sigma = sigma_pct / 100.0

# ─────────────────────────────────────────────────────────────
# Strategy presets
# ─────────────────────────────────────────────────────────────
PRESETS = {
    "Long Straddle": [
        {"type": "Call", "pos": "Long", "k": 1.00},
        {"type": "Put",  "pos": "Long", "k": 1.00},
    ],
    "Long Strangle": [
        {"type": "Call", "pos": "Long", "k": 1.05},
        {"type": "Put",  "pos": "Long", "k": 0.95},
    ],
    "Bull Call Spread": [
        {"type": "Call", "pos": "Long",  "k": 0.95},
        {"type": "Call", "pos": "Short", "k": 1.05},
    ],
    "Bear Put Spread": [
        {"type": "Put", "pos": "Long",  "k": 1.05},
        {"type": "Put", "pos": "Short", "k": 0.95},
    ],
    "Iron Condor": [
        {"type": "Put",  "pos": "Long",  "k": 0.90},
        {"type": "Put",  "pos": "Short", "k": 0.95},
        {"type": "Call", "pos": "Short", "k": 1.05},
        {"type": "Call", "pos": "Long",  "k": 1.10},
    ],
    "Long Call Butterfly": [
        {"type": "Call", "pos": "Long",  "k": 0.95, "qty": 1},
        {"type": "Call", "pos": "Short", "k": 1.00, "qty": 2},
        {"type": "Call", "pos": "Long",  "k": 1.05, "qty": 1},
    ],
}
LEG_COLUMNS = ["Type", "Position", "Strike", "Premium", "Qty"]


def price_leg(K, otype):
    """BSM premium for a single leg under current market parameters."""
    literal_type = "call" if str(otype).lower() == "call" else "put"
    try:
        return round(float(bsm_price_scalar(S, K, T, r, q, sigma, literal_type)), 4)
    except Exception:
        return 0.0


def build_preset_df(name):
    """Build the legs table for a preset; strikes scale with spot, premiums from BSM."""
    rows = []
    for leg in PRESETS[name]:
        K = round(S * leg["k"], 2)
        rows.append({
            "Type": leg["type"],
            "Position": leg["pos"],
            "Strike": K,
            "Premium": price_leg(K, leg["type"]),
            "Qty": leg.get("qty", 1),
        })
    return pd.DataFrame(rows, columns=LEG_COLUMNS).astype(
        {"Type": "object", "Position": "object", "Strike": "float64",
         "Premium": "float64", "Qty": "int64"})


def sanitize_legs(df):
    """Coerce the edited legs table into a clean, computable form."""
    d = pd.DataFrame(df).copy()
    for col in LEG_COLUMNS:
        if col not in d.columns:
            d[col] = None
    valid = d["Type"].isin(["Call", "Put"]) & d["Strike"].notna() & (d["Strike"] > 0)
    d = d.loc[valid].copy()
    d["Position"] = np.where(d["Position"].isin(["Long", "Short"]), d["Position"], "Long")
    d["Premium"] = pd.to_numeric(d["Premium"], errors="coerce").fillna(0.0).clip(lower=0.0)
    d["Qty"] = (pd.to_numeric(d["Qty"], errors="coerce").fillna(1.0)
                .clip(lower=1.0).round().astype("int64"))
    return d.reset_index(drop=True)


def auto_price_df(df):
    """Re-price every leg's premium with BSM under current market parameters."""
    d = sanitize_legs(df)
    d["Premium"] = [price_leg(float(K), t) for t, K in zip(d["Type"], d["Strike"])]
    return d


# ─────────────────────────────────────────────────────────────
# Session-state management — MUST run before the data_editor renders
# ─────────────────────────────────────────────────────────────
if "sb_legs_df" not in st.session_state:
    st.session_state["sb_legs_df"] = build_preset_df("Long Straddle")

# Deferred auto-price (button sets the flag, the following rerun consumes it
# AFTER the latest user edits were captured in sb_last_edited).
if st.session_state.pop("sb_autoprice_pending", False):
    st.session_state["sb_legs_df"] = auto_price_df(
        st.session_state.get("sb_last_edited", st.session_state["sb_legs_df"]))
    st.session_state.pop("sb_legs_editor", None)  # reset widget state

# ─────────────────────────────────────────────────────────────
# Layout — header
# ─────────────────────────────────────────────────────────────
st.markdown(
    f'<span class="live-badge">● BSM · σ={sigma:.0%} · T={T:.2f}y · r={r:.0%} · q={q:.0%}</span>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Preset picker + leg editor
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Strategy Presets</p>', unsafe_allow_html=True)

preset = st.selectbox(
    "Load a preset structure",
    list(PRESETS.keys()),
    help="Selecting a preset rebuilds the legs table below (strikes scale with spot, "
         "premiums are BSM-priced). Edit any leg afterwards — it becomes a custom strategy.",
)
if preset != st.session_state.get("sb_preset"):
    st.session_state["sb_preset"] = preset
    st.session_state["sb_legs_df"] = build_preset_df(preset)
    st.session_state.pop("sb_legs_editor", None)  # reset widget state

st.markdown('<p class="section-header">Strategy Legs</p>', unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1.4, 1.4, 2.2])
with col_btn1:
    auto_btn = st.button("⚡ Auto-Price Legs", width="stretch",
                         help="Recompute every leg's premium with Black-Scholes-Merton "
                              "under the current market parameters (S, r, q, T, σ).")
with col_btn2:
    reapply_btn = st.button("↺ Re-Apply Preset", width="stretch",
                            help="Discard manual edits and rebuild the selected preset.")

if auto_btn:
    st.session_state["sb_autoprice_pending"] = True
    st.rerun()
if reapply_btn:
    st.session_state["sb_legs_df"] = build_preset_df(preset)
    st.session_state.pop("sb_legs_editor", None)  # widget not yet instantiated this run

edited = st.data_editor(
    st.session_state["sb_legs_df"],
    key="sb_legs_editor",
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_order=LEG_COLUMNS,
    column_config={
        "Type": st.column_config.SelectboxColumn(
            "Type", options=["Call", "Put"], required=True,
            help="Call or Put."),
        "Position": st.column_config.SelectboxColumn(
            "Position", options=["Long", "Short"], required=True,
            help="Long = buy the option (pay premium); Short = write it (receive premium)."),
        "Strike": st.column_config.NumberColumn(
            "Strike K (₺)", min_value=0.01, step=0.5, format="%.2f",
            help="Strike price of the option."),
        "Premium": st.column_config.NumberColumn(
            "Premium (₺)", min_value=0.0, step=0.01, format="%.4f",
            help="Leg premium. Auto-filled from BSM; edit manually to override."),
        "Qty": st.column_config.NumberColumn(
            "Qty", min_value=1, step=1, format="%d",
            help="Number of contracts."),
    },
)
st.session_state["sb_last_edited"] = edited.copy()
st.caption("Add / remove rows as needed (2–6 legs). Premiums are editable — "
           "click **⚡ Auto-Price Legs** to refresh them from BSM.")

# ─────────────────────────────────────────────────────────────
# Legs validation
# ─────────────────────────────────────────────────────────────
legs = sanitize_legs(edited)
n_incomplete = len(edited) - len(legs)

if legs.empty:
    st.error("No valid legs defined. Add 2–6 legs (each needs a Type and a positive Strike).")
    st.stop()
if n_incomplete > 0:
    st.warning(f"{n_incomplete} incomplete leg(s) ignored (missing Type or Strike).")
if len(legs) == 1:
    st.info("Single-leg position defined — preset structures typically use 2–6 legs.")
elif len(legs) > 6:
    st.warning(f"{len(legs)} legs defined — the builder targets 2–6, but all legs are included.")

st.markdown(f"**{st.session_state.get('sb_preset', preset)}** · {len(legs)} leg(s) · "
            "see risk metrics below")

# ─────────────────────────────────────────────────────────────
# Payoff computation — vectorised over the spot grid
# ─────────────────────────────────────────────────────────────
N_PTS = 400
spot = np.linspace(0.5 * S, 1.5 * S, N_PTS)          # plot window: 0.5S → 1.5S


def compute_pnl(x, legs_df):
    """Aggregate expiry P&L over a spot array: per-leg option_payoff × Qty, summed.
    option_payoff already nets each leg's premium (subtract for long, add for short),
    so the sum equals aggregate intrinsic payoff minus net premium paid."""
    total = np.zeros_like(np.asarray(x, dtype=float))
    per_leg = []
    for leg in legs_df.itertuples():
        otype = "call" if leg.Type == "Call" else "put"
        pos = "long" if leg.Position == "Long" else "short"
        arr = option_payoff(np.asarray(x, dtype=float), float(leg.Strike),
                            float(leg.Premium), otype, pos) * int(leg.Qty)
        per_leg.append(arr)
        total = total + arr
    return total, per_leg


total_pnl, per_leg_pnl = compute_pnl(spot, legs)

# Net premium: Σ (long premiums paid) − Σ (short premiums received), × quantity
net_premium = float(sum(
    int(row.Qty) * (float(row.Premium) if row.Position == "Long" else -float(row.Premium))
    for row in legs.itertuples()
))
n_long = int((legs["Position"] == "Long").sum())
n_short = int((legs["Position"] == "Short").sum())

# ─────────────────────────────────────────────────────────────
# Risk metrics — computed numerically over an extended payoff grid
# (piecewise-linear in S_T ⇒ kinks only at strikes; grid spans 0 → 2S
#  plus every strike, so extremes and breakevens are exact)
# ─────────────────────────────────────────────────────────────
strikes = legs["Strike"].to_numpy(dtype=float)
mgrid = np.unique(np.concatenate([
    np.linspace(0.0, 2.0 * S, 601),
    strikes,
    np.array([S], dtype=float),
]))
mtotal, _ = compute_pnl(mgrid, legs)

slope_end = (mtotal[-1] - mtotal[-2]) / (mgrid[-1] - mgrid[-2])   # asymptotic P&L slope
EPS = 1e-12
max_profit_unbounded = slope_end > EPS     # net long calls as S_T → ∞
max_loss_unbounded = slope_end < -EPS      # net short calls as S_T → ∞
max_profit = float(mtotal.max())
max_loss = float(mtotal.min())


def extreme_segments(x, y, val, max_segments=3):
    """Human-readable S_T segments where P&L attains its extreme.
    Flat plateaus (spread/condor tails) are reported as ranges instead of an
    arbitrary float-dust argmin/argmax location."""
    tol = 1e-9 * max(1.0, abs(val))
    idxs = np.flatnonzero(np.abs(y - val) <= tol)
    if len(idxs) == 0:
        return [f"S_T = ₺{float(x[0]):,.2f}"]
    runs, start = [], idxs[0]
    for a, b in zip(idxs, idxs[1:]):
        if b - a > 1:
            runs.append((start, a))
            start = b
    runs.append((start, idxs[-1]))
    grid_lo, grid_hi = float(x[0]), float(x[-1])
    segs = []
    for i0, i1 in runs:
        lo, hi = float(x[i0]), float(x[i1])
        if lo <= grid_lo + 1e-12 and hi < grid_hi - 1e-12:
            segs.append(f"S_T ≤ ₺{hi:,.2f}")
        elif hi >= grid_hi - 1e-12 and lo > grid_lo + 1e-12:
            segs.append(f"S_T ≥ ₺{lo:,.2f}")
        elif lo == hi:
            segs.append(f"S_T = ₺{lo:,.2f}")
        else:
            segs.append(f"₺{lo:,.2f} ≤ S_T ≤ ₺{hi:,.2f}")
    if len(segs) > max_segments:
        segs = segs[:max_segments] + ["…"]
    return segs


def find_breakevens(x, y, tol):
    """Linear-interpolated zero crossings of the P&L curve."""
    bes = []
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if y1 == 0.0:
            bes.append(float(x[i]))
        elif y1 * y2 < 0.0:
            bes.append(float(x[i] - y1 * (x[i + 1] - x[i]) / (y2 - y1)))
    if len(y) and y[-1] == 0.0:
        bes.append(float(x[-1]))
    bes.sort()
    out = []
    for b in bes:
        if not out or abs(b - out[-1]) > tol:
            out.append(b)
    return out


breakevens = find_breakevens(mgrid, mtotal, tol=max(S * 1e-3, 1e-9))

# ─────────────────────────────────────────────────────────────
# Metrics row — dark metric cards
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Risk Metrics</p>', unsafe_allow_html=True)

if net_premium >= 0:
    prem_value, prem_sub, prem_cls = f"₺{net_premium:,.4f}", "Net Debit (cost)", "greek-neg"
else:
    prem_value, prem_sub, prem_cls = f"+₺{abs(net_premium):,.4f}", "Net Credit (received)", "greek-pos"

if max_profit_unbounded:
    profit_value, profit_sub = "Unlimited", "as S_T → ∞ (net long calls)"
else:
    profit_value = f"₺{max_profit:,.2f}"
    profit_sub = "for " + " or ".join(extreme_segments(mgrid, mtotal, max_profit))

if max_loss_unbounded:
    loss_value, loss_sub = "Unlimited", "as S_T → ∞ (net short calls)"
else:
    loss_value = f"₺{abs(max_loss):,.2f}"
    loss_sub = "for " + " or ".join(extreme_segments(mgrid, mtotal, max_loss))

if not breakevens:
    be_value, be_sub = "—", "no zero crossing in S_T ∈ [0, 2S]"
else:
    shown = [f"₺{b:,.2f}" for b in breakevens[:3]]
    be_value = "  ·  ".join(shown) + (f"  (+{len(breakevens) - 3} more)" if len(breakevens) > 3 else "")
    be_sub = f"{len(breakevens)} breakeven point(s)"

metric_cols = st.columns(4)
metrics = [
    ("Net Premium", prem_value, f"{prem_sub} · {n_long} long / {n_short} short", prem_cls),
    ("Max Profit",  profit_value, profit_sub, "greek-pos"),
    ("Max Loss",    loss_value,  loss_sub,    "greek-neg"),
    ("Breakeven(s)", be_value,   be_sub,      ""),
]
for col, (label, value, sub, cls) in zip(metric_cols, metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Payoff chart
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Payoff at Expiry</p>', unsafe_allow_html=True)
show_legs = st.toggle("Show per-leg P&L", value=False,
                      help="Overlay each leg's individual expiry P&L (premium included).")

LEG_COLORS = ["#fbbf24", "#a78bfa", "#34d399", "#f87171", "#60a5fa", "#f472b6"]

fig = go.Figure()

# Shaded profit / loss regions (fills to the zero line)
fig.add_trace(go.Scatter(
    x=spot, y=np.maximum(total_pnl, 0.0), mode="lines",
    line=dict(width=0), fill="tozeroy", fillcolor="rgba(52,211,153,0.20)",
    hoverinfo="skip", showlegend=False, name="Profit region"))
fig.add_trace(go.Scatter(
    x=spot, y=np.minimum(total_pnl, 0.0), mode="lines",
    line=dict(width=0), fill="tozeroy", fillcolor="rgba(248,113,113,0.20)",
    hoverinfo="skip", showlegend=False, name="Loss region"))

# Per-leg P&L overlays
if show_legs:
    for i, (leg, arr) in enumerate(zip(legs.itertuples(), per_leg_pnl)):
        fig.add_trace(go.Scatter(
            x=spot, y=arr, mode="lines",
            name=f"{leg.Position} {leg.Qty}× {leg.Type} K={leg.Strike:,.2f}",
            line=dict(color=LEG_COLORS[i % len(LEG_COLORS)], width=1.5, dash="dash"),
            hovertemplate="S_T=%{x:,.2f}<br>leg P&L=₺%{y:,.2f}<extra></extra>"))

# Total P&L line
fig.add_trace(go.Scatter(
    x=spot, y=total_pnl, mode="lines", name="Total P&L",
    line=dict(color="#38bdf8", width=3),
    hovertemplate="S_T=%{x:,.2f}<br>Total P&L=₺%{y:,.2f}<extra></extra>"))

# Zero line
fig.add_hline(y=0.0, line=dict(color="#94a3b8", width=1.5))

# Strike markers (vertical dotted lines + diamonds on the curve)
win_lo, win_hi = float(spot[0]), float(spot[-1])
plot_strikes = sorted({float(k) for k in strikes if win_lo <= k <= win_hi})
for k in plot_strikes:
    fig.add_vline(x=k, line=dict(color="#64748b", width=1.0, dash="dot"))
if plot_strikes:
    fig.add_trace(go.Scatter(
        x=plot_strikes, y=[float(np.interp(k, spot, total_pnl)) for k in plot_strikes],
        mode="markers", name="Strikes",
        marker=dict(symbol="diamond", size=10, color="#cbd5e1",
                    line=dict(width=1, color="#475569")),
        hovertemplate="K=%{x:,.2f}<extra></extra>"))

# Current spot marker
fig.add_trace(go.Scatter(
    x=[S], y=[float(np.interp(S, spot, total_pnl))], mode="markers", name="Spot",
    marker=dict(symbol="star", size=16, color="#f472b6",
                line=dict(width=1, color="#fbcfe8")),
    hovertemplate="Spot S=%{x:,.2f}<extra></extra>"))

# Breakeven vertical dashed lines
for be in breakevens:
    if win_lo <= be <= win_hi:
        fig.add_vline(x=be, line=dict(color="#fbbf24", width=1.3, dash="dash"), opacity=0.9)

fig.update_layout(
    template="plotly_dark",
    height=520,
    hovermode="x unified",
    xaxis_title="Spot at expiry S_T (₺)",
    yaxis_title="P&L at expiry (₺)",
    xaxis=dict(range=[win_lo, win_hi]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, width="stretch")

st.caption(
    f"Payoff evaluated over S_T ∈ [₺{win_lo:,.1f}, ₺{win_hi:,.1f}] ({N_PTS} points). "
    "Dotted grey lines: strikes · dashed amber lines: breakevens · pink star: current spot. "
    "Max profit / loss scanned numerically over S_T ∈ [0, 2S] including all strikes."
)
