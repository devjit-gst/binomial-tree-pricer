/* ==========================================================================
   Binomial Tree Pricer — front-end controller
   --------------------------------------------------------------------------
   Collects inputs, calls /api/price, and renders the returned Plotly figures
   and numbers. All numerics happen on the server; this file is presentation.
   ========================================================================== */

const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: false,
};

// Track the state of the two segmented ("toggle") controls.
const choice = { option_type: "call", exercise: "european" };

/* ----------------------------- Segmented UI ------------------------------ */
document.querySelectorAll(".segmented").forEach((group) => {
  const key = group.dataset.group;
  group.querySelectorAll(".seg").forEach((btn) => {
    btn.addEventListener("click", () => {
      group.querySelectorAll(".seg").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      choice[key] = btn.dataset.value;
    });
  });
});

/* ------------------------------- Helpers --------------------------------- */
const $ = (id) => document.getElementById(id);
const fmt = (x, dp = 4) =>
  Number.isFinite(x) ? x.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp }) : "—";

function showLoading(on) {
  $("loading").classList.toggle("hidden", !on);
}

function typesetMath() {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
}

/* ------------------------------ Rendering -------------------------------- */
function render(data) {
  // Summary cards.
  $("binom-price").textContent = fmt(data.binom_price);
  $("bs-price").textContent = fmt(data.bs_price);
  $("abs-diff").textContent = fmt(data.abs_diff, 5);
  $("card-N").textContent = `(N = ${$("N").value})`;

  // Greeks.
  const g = data.greeks;
  $("g-delta").textContent = fmt(g.delta, 4);
  $("g-gamma").textContent = fmt(g.gamma, 5);
  $("g-vega").textContent = fmt(g.vega, 4);
  $("g-theta").textContent = fmt(g.theta, 4);
  $("g-rho").textContent = fmt(g.rho, 4);

  // Risk-neutral panel.
  const rn = data.risk_neutral;
  $("rn-N").textContent = rn.draw_N;
  $("rn-dt").textContent = fmt(rn.dt, 5);
  $("rn-u").textContent = fmt(rn.u, 5);
  $("rn-d").textContent = fmt(rn.d, 5);
  $("rn-p").textContent = fmt(rn.p, 5);
  $("rn-disc").textContent = fmt(rn.disc_step, 5);
  $("rn-disct").textContent = fmt(rn.disc_total, 5);
  $("rn-exp").textContent = fmt(rn.expected_payoff, 4);
  $("rn-dexp").textContent = fmt(rn.discounted_expected_payoff, 4);

  // Plots.
  Plotly.react("lattice", data.lattice_fig.data, data.lattice_fig.layout, PLOTLY_CONFIG);
  Plotly.react("convergence", data.conv_fig.data, data.conv_fig.layout, PLOTLY_CONFIG);

  typesetMath();
}

/* ----------------------------- API request ------------------------------ */
async function priceOption() {
  const payload = {
    S: $("S").value,
    K: $("K").value,
    r: $("r").value,
    sigma: $("sigma").value,
    T: $("T").value,
    N: $("N").value,
    n_max: $("n_max").value,
    option_type: choice.option_type,
    exercise: choice.exercise,
  };

  $("err").textContent = "";
  showLoading(true);

  try {
    const resp = await fetch("/api/price", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `Request failed (${resp.status})`);
    }
    render(data);
  } catch (e) {
    $("err").textContent = e.message;
  } finally {
    showLoading(false);
  }
}

/* ------------------------------- Wire-up --------------------------------- */
$("pricer-form").addEventListener("submit", (e) => {
  e.preventDefault();
  priceOption();
});

// Price once on load so the dashboard is never empty.
window.addEventListener("load", priceOption);
