"""
QuantSuite Pricing Engine — Phase 1: Multi-Asset Engine
========================================================
Implements:
  - Black-Scholes-Merton (BSM) for dividend-paying equities
  - Garman-Kohlhagen (GK) for FX options
  - Newton-Raphson IV Solver with bisection fallback
"""
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class EquityMarketEnv:
    """Market environment for dividend-paying equity options (BSM)."""
    S: float        # Spot price
    K: float        # Strike price
    T: float        # Time to maturity in years
    r: float        # Domestic risk-free rate (annualised, continuous)
    q: float        # Continuous dividend yield
    sigma: float    # Annualised volatility


@dataclass
class FXMarketEnv:
    """Market environment for FX options (Garman-Kohlhagen)."""
    S: float        # Spot FX rate (domestic per foreign unit, e.g. TRY/USD)
    K: float        # Strike price
    T: float        # Time to maturity in years
    r_d: float      # Domestic risk-free rate (annualised, continuous)
    r_f: float      # Foreign risk-free rate (annualised, continuous)
    sigma: float    # Annualised volatility


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1% move in vol
    rho: float    # per 1% move in rate


@dataclass
class PricingResult:
    price: float
    greeks: Greeks
    d1: float
    d2: float


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """
    Compute d1 and d2 for any log-normal model.
    For GK: pass r=r_d, q=r_f.
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


# ─────────────────────────────────────────────────────────────
# Black-Scholes-Merton (Equity with continuous dividend yield)
# ─────────────────────────────────────────────────────────────

def bsm_price(env: EquityMarketEnv, option_type: Literal["call", "put"]) -> PricingResult:
    """
    Price a European equity option with continuous dividend yield using
    the Merton (1973) extension of Black-Scholes.
    """
    S, K, T, r, q, sigma = env.S, env.K, env.T, env.r, env.q, env.sigma
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_T = np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt_T)
    vega  = S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T / 100

    # Theta (per calendar day, using 365)
    theta_annual = (
        -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt_T)
        + q * S * np.exp(-q * T) * (norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1))
        - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else -norm.cdf(-d2))
    )
    theta = theta_annual / 365

    greeks = Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
    return PricingResult(price=price, greeks=greeks, d1=d1, d2=d2)


def bsm_price_scalar(S, K, T, r, q, sigma, option_type: Literal["call", "put"]) -> float:
    """Convenience wrapper returning only the premium (scalar)."""
    env = EquityMarketEnv(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    return bsm_price(env, option_type).price


# ─────────────────────────────────────────────────────────────
# Garman-Kohlhagen (FX options)
# ─────────────────────────────────────────────────────────────

def gk_price(env: FXMarketEnv, option_type: Literal["call", "put"]) -> PricingResult:
    """
    Price a European FX option using the Garman-Kohlhagen (1983) model.
    GK is structurally identical to BSM-Merton with q → r_f.
    """
    S, K, T, r_d, r_f, sigma = env.S, env.K, env.T, env.r_d, env.r_f, env.sigma
    d1, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    sqrt_T = np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
        delta = np.exp(-r_f * T) * norm.cdf(d1)
        rho   = K * T * np.exp(-r_d * T) * norm.cdf(d2) / 100
    else:
        price = K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)
        delta = -np.exp(-r_f * T) * norm.cdf(-d1)
        rho   = -K * T * np.exp(-r_d * T) * norm.cdf(-d2) / 100

    gamma = np.exp(-r_f * T) * norm.pdf(d1) / (S * sigma * sqrt_T)
    vega  = S * np.exp(-r_f * T) * norm.pdf(d1) * sqrt_T / 100

    theta_annual = (
        -S * np.exp(-r_f * T) * norm.pdf(d1) * sigma / (2 * sqrt_T)
        + r_f * S * np.exp(-r_f * T) * (norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1))
        - r_d * K * np.exp(-r_d * T) * (norm.cdf(d2) if option_type == "call" else -norm.cdf(-d2))
    )
    theta = theta_annual / 365

    greeks = Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
    return PricingResult(price=price, greeks=greeks, d1=d1, d2=d2)


def gk_price_scalar(S, K, T, r_d, r_f, sigma, option_type: Literal["call", "put"]) -> float:
    """Convenience wrapper returning only the premium (scalar)."""
    env = FXMarketEnv(S=S, K=K, T=T, r_d=r_d, r_f=r_f, sigma=sigma)
    return gk_price(env, option_type).price


# ─────────────────────────────────────────────────────────────
# IV Solver — Newton-Raphson with Bisection Fallback
# ─────────────────────────────────────────────────────────────

_IV_TOL = 1e-6
_IV_MAX_ITER = 200
_SIGMA_LOWER = 1e-6   # 0.0001% floor
_SIGMA_UPPER = 20.0   # 2000% ceiling


def _model_price(sigma: float, S, K, T, r, q, option_type, model: str) -> float:
    """Dispatch to correct pricing function given sigma."""
    if model == "bsm":
        return bsm_price_scalar(S, K, T, r, q, sigma, option_type)
    else:  # gk — r=r_d, q=r_f
        return gk_price_scalar(S, K, T, r, q, sigma, option_type)


def _vega_bsm(sigma: float, S, K, T, r, q) -> float:
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def _vega_gk(sigma: float, S, K, T, r_d, r_f) -> float:
    d1, _ = _d1_d2(S, K, T, r_d, r_f, sigma)
    return S * np.exp(-r_f * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,  # dividend yield for BSM, foreign rate for GK
    option_type: Literal["call", "put"],
    model: Literal["bsm", "gk"] = "bsm",
    initial_guess: float = 0.20,
) -> float:
    """
    Compute Implied Volatility using Newton-Raphson with bisection fallback.

    Parameters
    ----------
    market_price : float   Observed market premium
    S            : float   Spot price
    K            : float   Strike price
    T            : float   Time to maturity (years)
    r            : float   Risk-free rate (or domestic rate for GK)
    q            : float   Dividend yield (or foreign rate for GK)
    option_type  : str     'call' or 'put'
    model        : str     'bsm' or 'gk'
    initial_guess: float   Starting sigma for NR iterations

    Returns
    -------
    float : Implied Volatility (annualised)
    Raises ValueError if solver fails to converge.
    """
    if T <= 0:
        raise ValueError("Time to maturity must be positive.")
    if market_price <= 0:
        raise ValueError("Market price must be positive.")

    # Intrinsic value check
    if option_type == "call":
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    if market_price <= intrinsic:
        raise ValueError(
            f"Market price {market_price:.4f} ≤ intrinsic value {intrinsic:.4f}. "
            "No real implied volatility exists."
        )

    sigma = initial_guess

    # ── Newton-Raphson phase ──────────────────────────────────
    for _ in range(_IV_MAX_ITER):
        price = _model_price(sigma, S, K, T, r, q, option_type, model)
        diff = price - market_price

        if abs(diff) < _IV_TOL:
            return sigma

        if model == "bsm":
            vega = _vega_bsm(sigma, S, K, T, r, q)
        else:
            vega = _vega_gk(sigma, S, K, T, r, q)

        if vega < 1e-10:
            break  # degenerate vega → switch to bisection

        sigma -= diff / vega
        sigma = np.clip(sigma, _SIGMA_LOWER, _SIGMA_UPPER)

    # ── Bisection fallback ────────────────────────────────────
    lo, hi = _SIGMA_LOWER, _SIGMA_UPPER
    for _ in range(_IV_MAX_ITER):
        mid = 0.5 * (lo + hi)
        price = _model_price(mid, S, K, T, r, q, option_type, model)
        diff = price - market_price
        if abs(diff) < _IV_TOL or (hi - lo) < _IV_TOL:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid

    raise ValueError("IV solver failed to converge. Check inputs.")


# ─────────────────────────────────────────────────────────────
# Payoff profiles for strategy visualisation
# ─────────────────────────────────────────────────────────────

def option_payoff(spot_range: np.ndarray, K: float, premium: float,
                  option_type: Literal["call", "put"],
                  position: Literal["long", "short"] = "long") -> np.ndarray:
    """Return net P&L at expiry over a range of spot prices."""
    if option_type == "call":
        intrinsic = np.maximum(spot_range - K, 0)
    else:
        intrinsic = np.maximum(K - spot_range, 0)
    pnl = intrinsic - premium
    return pnl if position == "long" else -pnl
