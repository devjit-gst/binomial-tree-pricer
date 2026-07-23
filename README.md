# Binomial Tree Pricer

**Quant Researcher · Intro level**

A beautiful, interactive Flask dashboard that teaches **risk-neutral pricing**
from the ground up. It builds the Cox–Ross–Rubinstein (CRR) binomial tree,
prices European and American vanilla options by backward induction, and shows —
visually — how the binomial price **converges to Black–Scholes** as the number
of steps grows. This is the conceptual on-ramp to no-arbitrage / risk-neutral
valuation.

![lattice + convergence dashboard](https://img.shields.io/badge/Flask-dashboard-6ea8fe)

---

## Why this project exists

Almost every derivatives-pricing idea a quant researcher uses — martingale
measures, no-arbitrage, replication, PDE/expectation duality — can be seen in
miniature in a two-state (binomial) world. The binomial tree is the smallest
model where "the price is a *discounted expected payoff under a special
probability measure*" becomes concrete and computable. Once you internalise it,
Black–Scholes is just the continuous-time limit.

---

## The mathematics

### 1. No-arbitrage and the risk-neutral measure

Consider one time step in which the stock either goes up to \(S u\) or down to
\(S d\). Any option payoff over that step can be **replicated** by a portfolio
of \(\Delta\) shares of stock plus a cash position in the risk-free bond.
Because two portfolios with identical future payoffs must cost the same today
(otherwise there is an arbitrage), the option's price equals the cost of the
replicating portfolio.

Working through the algebra, that replication cost can be rewritten as a
**discounted expectation** — but under a synthetic probability, not the real
one:

$$
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\big[\text{payoff}(S_T)\big].
$$

Under the *risk-neutral measure* \(\mathbb{Q}\), every asset earns the risk-free
rate on average. Crucially, the **real-world probability of an up move never
appears** in the option price. Only volatility (which sets \(u\) and \(d\)) and
the risk-free rate matter.

### 2. The CRR parameterisation

Split maturity \(T\) into \(N\) steps of length \(\Delta t = T/N\). Cox, Ross
and Rubinstein chose the up/down factors to match the stock's volatility:

$$
u = e^{\sigma\sqrt{\Delta t}}, \qquad
d = \frac{1}{u} = e^{-\sigma\sqrt{\Delta t}}.
$$

The condition \(u\,d = 1\) makes the tree **recombine**: an up-then-down move
lands back where you started. So after \(i\) steps there are only \(i+1\)
distinct prices,

$$
S_i^{\,j} = S_0\, u^{\,j} d^{\,i-j}, \qquad j = 0,1,\dots,i,
$$

instead of \(2^i\). This is what keeps the model tractable.

The **risk-neutral up-probability** is pinned down by requiring the (discounted)
stock price to be a martingale — i.e. the stock earns exactly \(r\) on average:

$$
p\,u + (1-p)\,d = e^{r\Delta t}
\quad\Longrightarrow\quad
p = \frac{e^{r\Delta t} - d}{u - d}.
$$

For the model to be arbitrage-free we need \(d < e^{r\Delta t} < u\), which
guarantees \(0 < p < 1\).

### 3. Backward induction

Set the option's value at the terminal nodes to its payoff, then roll backwards.
At each node the value is the discounted risk-neutral expectation of its two
children:

$$
V_i^{\,j} = e^{-r\Delta t}\Big(p\,V_{i+1}^{\,j+1} + (1-p)\,V_{i+1}^{\,j}\Big).
$$

Repeating this from the leaves to the root gives today's price \(V_0^0\). Each
local step *is* the one-period risk-neutral pricing formula; chaining them is
just the law of iterated expectations.

### 4. American early exercise

An American option can be exercised at any node. So at every node we compare the
**continuation value** (holding) against the **intrinsic value** (exercising
now) and keep the larger:

$$
V_i^{\,j} = \max\!\Big(
\text{payoff}(S_i^{\,j}),\;
e^{-r\Delta t}\big(p\,V_{i+1}^{\,j+1} + (1-p)\,V_{i+1}^{\,j}\big)
\Big).
$$

Two classic results fall out of this and are checked in the smoke test:

- An **American call on a non-dividend stock equals the European call** — it is
  never optimal to exercise early (you would throw away time value and the
  interest on the strike).
- An **American put is worth strictly more** than the European put, because
  early exercise can be optimal when the option is deep in the money.

The dashboard rings the nodes where early exercise is optimal in red.

### 5. Convergence to Black–Scholes, and the sawtooth

As \(N \to \infty\), the CRR European price converges to the Black–Scholes
closed form. The error shrinks like \(\mathcal{O}(1/N)\), but it does **not**
decrease monotonically — it **oscillates**. This is the "money shot" of the
dashboard.

**Why the sawtooth?** For a fixed strike \(K\), the pricing error is dominated by
how the \(N+1\) discrete terminal nodes straddle the strike. As \(N\) increases
by one, the grid of terminal prices shifts, and whether a node lands just above
or just below \(K\) flips between "even \(N\)" and "odd \(N\)". That flips how
much probability mass sits just in- vs. just out-of-the-money near the money,
producing an even/odd oscillation whose amplitude decays as the grid gets finer.
The Black–Scholes value is the smooth limit that the sawtooth wraps around.

---

## What's in the box

| File | Purpose |
| --- | --- |
| `pricing.py` | Pure pricing engine: CRR tree (fast + full-lattice variants), Black–Scholes with greeks, convergence series. Heavily documented. |
| `app.py` | Flask server (port **5002**). Builds Plotly figures server-side and serves them as JSON to the front-end. |
| `templates/index.html` | Single-page dashboard with MathJax-rendered LaTeX. |
| `static/style.css` | "Research notebook" dark theme. |
| `static/app.js` | Front-end controller: gathers inputs, calls the API, renders results. |
| `smoke_test.py` | Headless correctness + boot checks. |
| `requirements.txt` | `flask`, `numpy`, `scipy`, `plotly`. |

### The three visualisations

1. **The recombining lattice** — drawn as a node graph (step on the x-axis, net
   up-moves on the y-axis). Nodes are coloured by option value; hover to see the
   asset price and option value at each node. American early-exercise nodes get
   a red ring.
2. **Convergence chart** — the binomial price as a function of \(N\), oscillating
   in a decaying sawtooth around the flat Black–Scholes benchmark line.
3. **Risk-neutral intuition panel** — displays \(\Delta t, u, d, p\), the step
   and total discount factors, and shows explicitly that the *discounted
   expected payoff under \(\mathbb{Q}\)* reproduces the price.

---

## How to run

From a PowerShell prompt in this folder:

```powershell
# 1. (optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt

# 3. run the headless smoke test (no server left running)
python smoke_test.py

# 4. launch the dashboard
python app.py
```

Then open <http://127.0.0.1:5002> in your browser.

The dashboard prices an at-the-money 1-year call on load. Adjust the inputs in
the left panel and click **Price option** to re-price and redraw everything.

> **Tip:** set the number of steps to a small value (e.g. `N = 5`) to see an
> individual lattice clearly, and push the "Convergence up to N" slider to ~200
> to watch the sawtooth settle onto the Black–Scholes line.

---

## Verifying correctness

`smoke_test.py` asserts, among other things, that for the standard case
\(S=K=100,\ r=5\%,\ \sigma=20\%,\ T=1\):

- the European binomial call price agrees with Black–Scholes to
  \(|\text{diff}| < 0.05\) at \(N = 500\);
- the American call equals the European call (no dividends);
- the American put exceeds the European put;
- Black–Scholes put–call parity holds;
- the Flask app boots and `GET /` returns HTTP 200.

---

## Notes

Built and deployed by Devjit.

*Educational use only — not investment advice.*
