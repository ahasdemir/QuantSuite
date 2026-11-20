# QuantSuite

**A multi-asset derivatives and volatility analytics platform — Black-Scholes-Merton and Garman-Kohlhagen pricing, full Greeks, a Newton-Raphson implied-volatility solver, 3D volatility surfaces and a multi-leg strategy simulator.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3D%20surfaces-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

QuantSuite is a quantitative toolkit for analysts and risk managers working with **equity and FX options**. It bridges academic theory and trading-desk practice: analytic pricing models, vectorised Greeks, numerical implied-volatility inversion, volatility-surface visualisation and expiry payoff simulation — all in an interactive Streamlit dashboard.

---

## Screenshots

### Equity Derivatives — BSM pricing, Greeks & IV solver
![Equity Derivatives](docs/screenshots/equity_derivatives.png)

### FX Derivatives — Garman-Kohlhagen dual-rate pricing
![FX Derivatives](docs/screenshots/fx_derivatives.png)

### Volatility Analysis — 3D parametric surface, smile & second-order Greeks
![Volatility Analysis](docs/screenshots/volatility_analysis.png)

### Strategy Builder — multi-leg payoff simulation & risk metrics
![Strategy Builder](docs/screenshots/strategy_builder.png)

### Home
![Home](docs/screenshots/home.png)

---

## Modules

| Page | What it does |
|---|---|
| **Equity Derivatives** | BSM pricing for dividend-paying equities; Delta, Gamma, Theta, Vega, Rho; IV solver; expiry-payoff and sensitivity tabs |
| **FX Derivatives** | Garman-Kohlhagen pricing with domestic/foreign rates; same Greeks suite and IV inversion |
| **Volatility Analysis** | Parametric 3D IV surface (strike × maturity), smile cross-sections at 1M/3M/6M, Gamma and Vanna heatmaps |
| **Strategy Builder** | Multi-leg (2–6) option strategies with BSM-priced legs, expiry payoff, breakevens and max profit/loss |
| **Home** | Overview and navigation |

---

## Key Capabilities

- **Hybrid pricing engine** — dividend-adjusted Black-Scholes-Merton for equities; Garman-Kohlhagen dual-interest-rate model for FX.
- **Greeks lab** — vectorised Delta, Gamma, Theta, Vega and Rho across single options and portfolios.
- **Implied-volatility solver** — Newton-Raphson with automatic bisection fallback for robust convergence near the wings and deep ITM/OTM.
- **Surface mapping** — 3D implied-volatility surfaces (maturity × strike × IV) that expose skew and term structure, with an explicit parametric-model disclaimer.
- **Volatility smile** — IV-vs-strike cross-sections at multiple maturities with ATM marker.
- **Second-order Greeks** — analytic Gamma and Vanna heatmaps over spot × maturity.
- **Strategy simulation** — six presets (long straddle, long strangle, bull call spread, bear put spread, iron condor, long call butterfly), editable legs, payoff chart with breakevens and "Unlimited" risk handling.

---

## Financial Methodology

### Black-Scholes-Merton (equities, continuous dividend yield q)

$$C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2), \qquad P = K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

### Garman-Kohlhagen (FX)

Same structure with $r \to r_d$ (domestic) and $q \to r_f$ (foreign):

$$C = S e^{-r_f T}N(d_1) - K e^{-r_d T}N(d_2)$$

### Greeks

$$\Delta_{call} = e^{-qT}N(d_1), \quad \Gamma = \frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}, \quad \mathcal{V} = S e^{-qT}\phi(d_1)\sqrt{T}$$

$$\Theta = -\frac{S e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1), \qquad \rho = KTe^{-rT}N(d_2)$$

### Implied volatility

Solve $\text{Price}_{model}(\sigma) = \text{Price}_{market}$ for $\sigma$:

$$\sigma_{n+1} = \sigma_n - \frac{f(\sigma_n)}{f'(\sigma_n)}, \qquad f(\sigma) = \text{BSM}(\sigma) - P_{mkt}$$

with bisection fallback when the NR step leaves the admissible region or stalls.

### Second-order cross-Greek

$$\text{Vanna} = \frac{\partial^2 V}{\partial S\,\partial \sigma} = -\frac{e^{-qT}\phi(d_1)d_2}{\sigma}$$

---

## Quick Start

```bash
git clone https://github.com/ahasdemir/QuantSuite.git
cd QuantSuite

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run streamlit_app.py
```

Open `http://localhost:8501` and pick a module from the sidebar.

**Tips**
- Equity page: toggle **Fetch Live Spot** off to price against a manual spot (useful offline).
- Volatility page: the surface is **parametric** (realised vol + adjustable skew/smile) — it is a model surface, not market quotes.
- Strategy Builder: use **Auto-Price Legs** after changing σ, T or spot; manual premium edits are preserved until you re-price.
- IV solver: enter a market premium to invert it; convergence failures raise explicit errors instead of returning silent garbage.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Numerical | `numpy`, `scipy` |
| Visualisation | Plotly (`go.Surface`, heatmaps, payoff charts) |
| App | Streamlit multi-page |
| Data | `yfinance` / `borsapy` for spot prices (optional, graceful fallback) |

---

## Project Structure

```
QuantSuite/
├── streamlit_app.py              # entry point
├── pricing_engine.py             # BSM, Garman-Kohlhagen, Greeks, IV solver
├── analysis_utils.py             # data layer (inherited from Algo-Risk Monitor)
├── pages/
│   ├── home.py
│   ├── equity_derivatives.py
│   ├── fx_derivatives.py
│   ├── volatility_analysis.py
│   └── strategy_builder.py
├── docs/screenshots/             # README screenshots
├── requirements.txt
└── README.md
```

---

## Roadmap

### Phase 1 — Multi-Asset Engine ✅
- [x] Garman-Kohlhagen for FX and Black-Scholes-Merton for dividend-paying equities
- [x] Newton-Raphson IV solver with bisection fallback
- [x] Standardised domestic/foreign rate input pipeline

### Phase 2 — Advanced Visuals & Surfaces ✅
- [x] Plotly 3D volatility surfaces (skew and term structure)
- [x] Volatility smile cross-sections (1M / 3M / 6M)
- [x] Gamma and Vanna sensitivity heatmaps

### Phase 3 — Trading Strategy & I/O ✅
- [x] Multi-leg strategy builder with BSM-priced legs and expiry P&L (6 presets)

### Next
- [ ] Live option-chain import (BIST / CBOE) to replace the parametric surface with market quotes
- [ ] Calibration engine (SVI / SABR) fitted to observed smiles
- [ ] Portfolio-level Greeks aggregation across mixed equity/FX books
- [ ] Exportable risk reports (PDF / Excel)

---

## Evolution & Credits

QuantSuite is an evolution of **[Algo-Risk Monitor](https://github.com/ahasdemir/Algo-Risk-Monitor)**. The core `analysis_utils.py` data-fetching layer was inherited and extended with derivative-specific mathematics, stochastic simulation and a dedicated pricing engine — a modular path from portfolio risk analytics to niche derivatives tooling.

---

## Author

**Ahmet Hasdemir** — Economics student (Math minor), Hacettepe University · quant finance & econometrics

[GitHub](https://github.com/ahasdemir) · [LinkedIn](https://www.linkedin.com/in/ahasdemir/)

## License

MIT — see [LICENSE](LICENSE).
