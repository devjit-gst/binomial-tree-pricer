"""
app.py
======

Flask dashboard for the *Binomial Tree Pricer*.

The server does all of the numerical heavy lifting (in :mod:`pricing`) and
constructs Plotly figures as JSON, which the front-end renders with Plotly.js.
Keeping figure construction on the server means the browser only ever handles
presentation, and the interesting mathematics lives in one documented place.

Run with:

    python app.py

then open http://127.0.0.1:5002 in a browser.
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, jsonify, render_template, request

import pricing

app = Flask(__name__)

# ---------------------------------------------------------------------------
# A calm "research notebook" colour palette shared across every figure.
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#0f1420",
    "panel": "#151b2b",
    "grid": "#26304a",
    "text": "#e6ecff",
    "muted": "#8a97b8",
    "accent": "#6ea8fe",   # binomial / primary
    "accent2": "#f5a97f",  # black-scholes / benchmark
    "up": "#63d2a4",       # up-moves / in-the-money
    "down": "#f28fad",     # down-moves
    "exercise": "#f7768e",
}


def _base_layout(title: str) -> dict:
    """Common Plotly layout so every chart shares the dashboard aesthetic."""
    return dict(
        title=dict(text=title, font=dict(size=16, color=PALETTE["text"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter, system-ui, sans-serif"),
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor=PALETTE["panel"],
            bordercolor=PALETTE["grid"],
            font=dict(color=PALETTE["text"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=PALETTE["muted"]),
        ),
    )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def build_lattice_figure(res: pricing.BinomialResult, option_type: str) -> go.Figure:
    """Draw the recombining CRR lattice as a node-graph.

    Node ``(i, j)`` (``i`` = step, ``j`` = up-moves) is placed at::

        x = i            (time / step index)
        y = 2*j - i      (net up-moves; keeps the tree symmetric & recombining)

    Edges connect each node to its up and down children. Nodes are coloured by
    option value; nodes where early exercise is optimal get a distinct ring.
    """
    N = res.n_steps

    # --- Edges (drawn first so nodes sit on top). ---------------------------
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for i in range(N):
        for j in range(i + 1):
            x0, y0 = i, 2 * j - i
            # Up child (j + 1) and down child (j) at step i + 1.
            for jj in (j + 1, j):
                x1, y1 = i + 1, 2 * jj - (i + 1)
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color=PALETTE["grid"], width=1),
        hoverinfo="skip",
        showlegend=False,
    )

    # --- Nodes. -------------------------------------------------------------
    node_x, node_y = [], []
    colors, texts, ring_x, ring_y = [], [], [], []
    for i in range(N + 1):
        for j in range(i + 1):
            x, y = i, 2 * j - i
            node_x.append(x)
            node_y.append(y)
            asset = res.asset_tree[i][j]
            value = res.value_tree[i][j]
            colors.append(value)
            texts.append(
                f"<b>Step {i}</b> &nbsp; up-moves {j}<br>"
                f"Asset price&nbsp;&nbsp;S = {asset:,.4f}<br>"
                f"Option value&nbsp; V = {value:,.4f}"
                + (
                    "<br><b>early exercise optimal</b>"
                    if res.exercise_tree[i][j] and i < N
                    else ""
                )
            )
            if res.exercise_tree[i][j] and i < N:
                ring_x.append(x)
                ring_y.append(y)

    # Sizing: shrink the markers as the tree grows so it stays readable.
    marker_size = max(6, min(22, 420 / (N + 2)))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=marker_size,
            color=colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text="Option<br>value", font=dict(color=PALETTE["muted"])),
                tickfont=dict(color=PALETTE["muted"]),
                outlinewidth=0,
                thickness=12,
            ),
            line=dict(color=PALETTE["bg"], width=1),
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    traces = [edge_trace, node_trace]

    # Highlight early-exercise nodes with a red ring (American options).
    if ring_x:
        traces.append(
            go.Scatter(
                x=ring_x,
                y=ring_y,
                mode="markers",
                marker=dict(
                    size=marker_size + 6,
                    color="rgba(0,0,0,0)",
                    line=dict(color=PALETTE["exercise"], width=2),
                ),
                name="early exercise",
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=traces)
    layout = _base_layout(f"Recombining CRR lattice&nbsp; ({option_type}, N = {N})")
    layout.update(
        xaxis=dict(
            title="step i",
            showgrid=False,
            zeroline=False,
            color=PALETTE["muted"],
        ),
        yaxis=dict(
            title="net up-moves (2j - i)",
            showgrid=False,
            zeroline=False,
            color=PALETTE["muted"],
        ),
        showlegend=bool(ring_x),
    )
    fig.update_layout(layout)
    return fig


def build_convergence_figure(
    steps: list[int],
    prices: list[float],
    bs_price: float,
    option_type: str,
) -> go.Figure:
    """Binomial price vs. N, converging to the flat Black-Scholes line."""
    fig = go.Figure()

    # The Black-Scholes benchmark as a horizontal reference line.
    fig.add_trace(
        go.Scatter(
            x=[steps[0], steps[-1]],
            y=[bs_price, bs_price],
            mode="lines",
            line=dict(color=PALETTE["accent2"], width=2, dash="dash"),
            name=f"Black-Scholes = {bs_price:.4f}",
        )
    )

    # The oscillating binomial price (the "sawtooth").
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=prices,
            mode="lines+markers",
            line=dict(color=PALETTE["accent"], width=1.5),
            marker=dict(size=3, color=PALETTE["accent"]),
            name="Binomial price",
            hovertemplate="N = %{x}<br>price = %{y:.4f}<extra></extra>",
        )
    )

    layout = _base_layout(
        f"Binomial → Black-Scholes convergence&nbsp; ({option_type})"
    )
    layout.update(
        xaxis=dict(
            title="number of steps N",
            gridcolor=PALETTE["grid"],
            zeroline=False,
            color=PALETTE["muted"],
        ),
        yaxis=dict(
            title="option price",
            gridcolor=PALETTE["grid"],
            zeroline=False,
            color=PALETTE["muted"],
        ),
    )
    fig.update_layout(layout)
    return fig


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the single-page dashboard."""
    return render_template("index.html")


@app.route("/api/price", methods=["POST"])
def api_price():
    """Price the option and return figures + numbers as JSON.

    Expects a JSON body with the pricing inputs. Returns:
      * scalar prices (binomial, Black-Scholes) and greeks,
      * risk-neutral quantities (u, d, p, discount factor, ...),
      * two Plotly figures (lattice + convergence) as JSON.
    """
    data = request.get_json(force=True)

    try:
        S = float(data["S"])
        K = float(data["K"])
        r = float(data["r"])
        sigma = float(data["sigma"])
        T = float(data["T"])
        N = int(data["N"])
        option_type = data.get("option_type", "call")
        exercise = data.get("exercise", "european")
        n_max = int(data.get("n_max", 200))
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify(error=f"Invalid input: {exc}"), 400

    # --- Validation. --------------------------------------------------------
    if min(S, K, sigma, T) <= 0:
        return jsonify(error="S, K, sigma and T must all be positive."), 400
    if not (1 <= N <= 400):
        return jsonify(error="N (lattice steps) must be between 1 and 400."), 400
    if not (1 <= n_max <= 1000):
        return jsonify(error="Convergence N must be between 1 and 1000."), 400

    # --- Core valuations. ---------------------------------------------------
    # Full lattice (retained) for the node-graph. Cap the drawn tree so the
    # visualisation stays legible even if the user asks for many steps.
    draw_N = min(N, 60)
    res = pricing.crr_tree(S, K, r, sigma, T, draw_N, option_type, exercise)

    # A "precise" binomial price at the user's requested N (may exceed draw_N).
    binom_price = pricing.crr_price(S, K, r, sigma, T, N, option_type, exercise)

    bs = pricing.black_scholes(S, K, r, sigma, T, option_type)

    steps, conv_prices = pricing.convergence_series(
        S, K, r, sigma, T, option_type, exercise, n_max=n_max
    )

    lattice_fig = build_lattice_figure(res, option_type)
    conv_fig = build_convergence_figure(steps, conv_prices, bs.price, option_type)

    # Expected discounted payoff decomposition for the intuition panel, using
    # the risk-neutral binomial distribution over terminal nodes at draw_N.
    terminal_spots = np.array(res.asset_tree[draw_N])
    terminal_payoffs = (
        np.maximum(terminal_spots - K, 0.0)
        if option_type == "call"
        else np.maximum(K - terminal_spots, 0.0)
    )
    j = np.arange(draw_N + 1)
    # Binomial probabilities under the risk-neutral measure.
    from math import comb

    rn_probs = np.array(
        [comb(draw_N, int(jj)) * res.p**jj * (1 - res.p) ** (draw_N - jj) for jj in j]
    )
    expected_payoff = float(np.sum(rn_probs * terminal_payoffs))
    total_discount = float(res.disc**draw_N)  # exp(-r T)

    payload = {
        "binom_price": binom_price,
        "binom_price_drawn": res.price,
        "bs_price": bs.price,
        "abs_diff": abs(binom_price - bs.price),
        "greeks": {
            "delta": bs.delta,
            "gamma": bs.gamma,
            "vega": bs.vega / 100.0,   # report per 1% vol move
            "theta": bs.theta / 365.0,  # report per calendar day
            "rho": bs.rho / 100.0,     # report per 1% rate move
            "d1": bs.d1,
            "d2": bs.d2,
        },
        "risk_neutral": {
            "dt": res.dt,
            "u": res.u,
            "d": res.d,
            "p": res.p,
            "disc_step": res.disc,
            "disc_total": total_discount,
            "draw_N": draw_N,
            "expected_payoff": expected_payoff,
            "discounted_expected_payoff": expected_payoff * total_discount,
        },
        "lattice_fig": json.loads(
            json.dumps(lattice_fig, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "conv_fig": json.loads(
            json.dumps(conv_fig, cls=plotly.utils.PlotlyJSONEncoder)
        ),
    }
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True, port=5002)
