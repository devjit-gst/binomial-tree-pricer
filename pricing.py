"""
pricing.py
==========

Pricing engine for the *Binomial Tree Pricer* educational project.

This module implements two classic approaches to pricing European/American
vanilla options and exposes everything needed to *visualise* the mechanics of
risk-neutral valuation:

1. The Cox-Ross-Rubinstein (CRR) binomial tree.
2. The Black-Scholes closed-form solution (European benchmark) with greeks.

--------------------------------------------------------------------------
Risk-neutral pricing in one paragraph
--------------------------------------------------------------------------
Under the *no-arbitrage* assumption we can construct a self-financing
portfolio of the underlying and a risk-free bond that perfectly replicates an
option's payoff over a small time step. Because the replicating portfolio and
the option must cost the same today (otherwise a free lunch exists), the
option price does **not** depend on the real-world probability of the stock
going up. Instead it equals the *discounted expectation of the payoff under a
special probability measure* -- the risk-neutral measure -- in which every
asset earns the risk-free rate on average. In the binomial world that measure
is captured by a single number ``p``.

--------------------------------------------------------------------------
The CRR parameterisation
--------------------------------------------------------------------------
Split the life of the option ``T`` into ``N`` steps of length ``dt = T / N``.
Over one step the stock either multiplies by ``u`` (up) or ``d`` (down):

    u = exp(sigma * sqrt(dt))          (up factor)
    d = 1 / u = exp(-sigma * sqrt(dt)) (down factor -> recombining tree)

The recombining property (``u * d = 1``) is what makes the lattice collapse
to ``N + 1`` terminal nodes instead of ``2**N`` -- essential for tractability.

The risk-neutral up-probability is pinned down by requiring the discounted
stock price to be a martingale (i.e. the stock earns ``r`` on average):

    E^Q[S_{t+dt}] = S_t * exp(r * dt)
    => p * u + (1 - p) * d = exp(r * dt)
    => p = (exp(r * dt) - d) / (u - d)

The option value at any node is the discounted risk-neutral expectation of its
two children:

    V = exp(-r * dt) * (p * V_up + (1 - p) * V_down)

For an **American** option we additionally compare this "continuation value"
against the payoff from exercising immediately and take the larger of the two.

As ``N -> infinity`` the binomial European price converges to Black-Scholes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class BinomialResult:
    """Full output of a binomial valuation, including the lattices."""

    price: float
    # CRR step parameters (useful for the risk-neutral intuition panel).
    dt: float
    u: float
    d: float
    p: float
    disc: float  # per-step discount factor exp(-r * dt)
    n_steps: int
    # Lattices stored as lists-of-lists so they serialise cleanly to JSON.
    # asset_tree[i][j] = underlying price after i steps with j up-moves.
    asset_tree: list[list[float]] = field(default_factory=list)
    # value_tree[i][j] = option value at that node.
    value_tree: list[list[float]] = field(default_factory=list)
    # exercise_tree[i][j] = True where early exercise is optimal (American).
    exercise_tree: list[list[bool]] = field(default_factory=list)


@dataclass
class BlackScholesResult:
    """Closed-form European price plus the standard greeks."""

    price: float
    delta: float
    gamma: float
    vega: float   # per 1.00 (i.e. 100%) change in vol; divide by 100 for 1%
    theta: float  # per year; divide by 365 for per-day
    rho: float    # per 1.00 change in r
    d1: float
    d2: float


# ---------------------------------------------------------------------------
# Black-Scholes closed form
# ---------------------------------------------------------------------------
def black_scholes(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType = "call",
) -> BlackScholesResult:
    """Price a European option with the Black-Scholes-Merton formula.

    Parameters
    ----------
    S : spot price of the underlying
    K : strike
    r : continuously-compounded risk-free rate
    sigma : annualised volatility
    T : time to maturity in years
    option_type : 'call' or 'put'

    Returns
    -------
    BlackScholesResult with price and greeks.

    Notes
    -----
    The formula is the ``N -> infinity`` limit of the CRR tree, so it is the
    natural benchmark against which we measure binomial convergence.
    """
    if T <= 0 or sigma <= 0:
        # Degenerate limit: value is the discounted intrinsic payoff.
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return BlackScholesResult(intrinsic, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * sqrt(T))
            - r * K * exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * sqrt(T))
            + r * K * exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * exp(-r * T) * norm.cdf(-d2)

    # Gamma and vega are identical for calls and puts.
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * norm.pdf(d1) * sqrt(T)

    return BlackScholesResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        d1=d1,
        d2=d2,
    )


# ---------------------------------------------------------------------------
# CRR binomial tree
# ---------------------------------------------------------------------------
def _payoff(spots: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    """Vanilla payoff evaluated element-wise on an array of spot prices."""
    if option_type == "call":
        return np.maximum(spots - K, 0.0)
    return np.maximum(K - spots, 0.0)


def crr_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "european",
) -> float:
    """Fast CRR price (no lattice retained) using vectorised backward induction.

    This is the workhorse used by the convergence study, where we re-price for
    every ``N`` from 1 to a few hundred and therefore care about speed. Only
    the option price is returned.
    """
    if N < 1:
        raise ValueError("N must be a positive integer.")

    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    disc = exp(-r * dt)
    p = (exp(r * dt) - d) / (u - d)

    # Terminal underlying prices: S * u**j * d**(N - j) for j = 0..N.
    j = np.arange(N + 1)
    spots = S * (u**j) * (d ** (N - j))
    values = _payoff(spots, K, option_type)

    # Walk backwards through the lattice, one time-slice at a time.
    for i in range(N - 1, -1, -1):
        # Continuation (discounted risk-neutral expectation of the children).
        values = disc * (p * values[1 : i + 2] + (1 - p) * values[0 : i + 1])
        if exercise == "american":
            spots = S * (u ** np.arange(i + 1)) * (d ** (np.arange(i + 1)[::-1]))
            values = np.maximum(values, _payoff(spots, K, option_type))

    return float(values[0])


def crr_tree(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "european",
) -> BinomialResult:
    """CRR price **with** the full asset and option-value lattices retained.

    Used for the interactive lattice visualisation, so ``N`` is expected to be
    modest (a few dozen at most). The returned lattices are triangular:
    ``tree[i]`` has ``i + 1`` entries, indexed by number of up-moves ``j``.

    Backward induction is done node-by-node (rather than vectorised) so that we
    can record, at each node, whether early exercise was optimal.
    """
    if N < 1:
        raise ValueError("N must be a positive integer.")

    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    disc = exp(-r * dt)
    p = (exp(r * dt) - d) / (u - d)

    # --- Forward pass: build the asset-price lattice. -----------------------
    asset_tree: list[list[float]] = []
    for i in range(N + 1):
        level = [S * (u**j) * (d ** (i - j)) for j in range(i + 1)]
        asset_tree.append(level)

    # --- Backward pass: option values and exercise decisions. ---------------
    # Initialise value/exercise lattices with the right (triangular) shape.
    value_tree: list[list[float]] = [[0.0] * (i + 1) for i in range(N + 1)]
    exercise_tree: list[list[bool]] = [[False] * (i + 1) for i in range(N + 1)]

    # Terminal payoffs.
    for j in range(N + 1):
        payoff = _intrinsic(asset_tree[N][j], K, option_type)
        value_tree[N][j] = payoff
        # At expiry, "exercising" and "holding" coincide when in the money.
        exercise_tree[N][j] = payoff > 0.0

    # Roll back through the tree.
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation = disc * (
                p * value_tree[i + 1][j + 1] + (1 - p) * value_tree[i + 1][j]
            )
            if exercise == "american":
                intrinsic = _intrinsic(asset_tree[i][j], K, option_type)
                if intrinsic > continuation:
                    value_tree[i][j] = intrinsic
                    exercise_tree[i][j] = True
                else:
                    value_tree[i][j] = continuation
            else:
                value_tree[i][j] = continuation

    return BinomialResult(
        price=value_tree[0][0],
        dt=dt,
        u=u,
        d=d,
        p=p,
        disc=disc,
        n_steps=N,
        asset_tree=asset_tree,
        value_tree=value_tree,
        exercise_tree=exercise_tree,
    )


def _intrinsic(spot: float, K: float, option_type: OptionType) -> float:
    """Scalar vanilla payoff (helper for the node-by-node backward pass)."""
    if option_type == "call":
        return max(spot - K, 0.0)
    return max(K - spot, 0.0)


def convergence_series(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "european",
    n_max: int = 200,
) -> tuple[list[int], list[float]]:
    """Binomial price as a function of the number of steps ``N``.

    Returns ``(steps, prices)`` where ``steps = [1, 2, ..., n_max]``. Plotting
    ``prices`` against ``steps`` produces the characteristic *sawtooth* that
    oscillates around and converges to the Black-Scholes value.

    Why the sawtooth? For a fixed strike, whether one of the ``N + 1`` terminal
    nodes lands close to ``K`` alternates as ``N`` increases by one. This shifts
    how much probability mass sits just in/out of the money near the strike,
    producing an even/odd oscillation whose amplitude decays like ``O(1/N)``.
    """
    steps = list(range(1, n_max + 1))
    prices = [
        crr_price(S, K, r, sigma, T, n, option_type, exercise) for n in steps
    ]
    return steps, prices


if __name__ == "__main__":
    # Quick self-check when run directly: European call should match BS.
    params = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
    bs = black_scholes(option_type="call", **params)
    binom = crr_price(N=500, option_type="call", exercise="european", **params)
    print(f"Black-Scholes call : {bs.price:.6f}")
    print(f"Binomial (N=500)   : {binom:.6f}")
    print(f"|difference|       : {abs(bs.price - binom):.6f}")
