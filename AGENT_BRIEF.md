# Brief for the research agent

You are working on the **Generalized WhiteBox Challenge**. Read `README.md` for the full
specification; this page is your mission and how to work.

## Mission

Find a **general** method that predicts the per-neuron mean activations of a random MLP from
its weights, within a FLOP budget, and that works across **all 8 activation functions, all 4
weight distributions, and all widths (16–384) and depths (4–24)** in the dataset — not a method
tuned to one activation or one shape.

The number that counts is the **test-split headline score** (geometric mean of the
compute-adjusted ratio to budget-matched sampling; lower is better; `1.0` = as good as
spending the whole budget on Monte-Carlo sampling). A method that is excellent on ReLU and
useless on `gabor` or `tanh_rmsnorm` will score badly; so will one that wins at width 384 and
loses at width 16. Look at the per-activation / per-strategy / per-width / per-depth breakdowns
and the worst-decile figure after every run — they tell you where your method does not
generalize.

## How to work

```bash
cd <repo> && uv sync
uv run gwc info                                   # what the dataset looks like
uv run gwc smoke  --estimator examples/cov_estimator.py    # seconds: 8 small networks
uv run gwc submit --estimator examples/cov_estimator.py    # ~minutes: all 256 train networks
```

1. Start from `examples/cov_estimator.py` (full-covariance Gaussian moment propagation with
   Gauss–Hermite quadrature — activation-agnostic) or `examples/gh_estimator.py` (mean-field).
   Understand *why* they lose to sampling on narrow/deep networks before trying to fix it.
2. Iterate with `gwc smoke` (fast) and `gwc submit` on **train** (unlimited). Read
   `runs/<run>/result.json` for every per-network row and `worker.log` for tracebacks.
3. Use the **test** split (`--split test`) sparingly — it is limited to **once every 4 hours** and
   is the score of record. Check `gwc status`. A large train→test gap means you over-fitted.
4. Keep a log of what you tried and what each change did to the breakdowns.

## Rules — non-negotiable

* **All arithmetic inside `predict` must go through `flopscope.numpy` (`fnp`) and
  `gwc.sdk.activation`.** Never use plain NumPy/SciPy/torch there; FLOPs must be metered or
  your compute discount is a lie. (`setup()` is off-budget; precompute tables there if useful.)
* **Do not read or use the ground truth, the encryption key, the cooldown state, or
  `data/cache/`.** Do not tamper with the cooldown. Do not special-case network indices, names or
  seeds. You are trusted not to cheat; the harness is not an adversarial sandbox.
* Do not simulate the network with unmetered code to "estimate" the answer. Sampling *is*
  allowed if metered (see `examples/mc_estimator.py`) — it is the baseline you must beat.
* Never re-seed from `net.seed` directly for sampling: that stream produced the weights. Derive
  an independent stream (`fnp.random.SeedSequence([net.seed, 1])`).

## What "general" means here (hints, not answers)

* The activations were chosen to be pairwise non-redundant as *moment maps* — there is no
  closed-form trick that covers them all. Numerical quadrature on Gaussian marginals is
  general; the question is what to do about the non-Gaussianity and inter-neuron correlations
  that appear at small width and large depth.
* Weight distributions differ in shape, not scale: `orth` is exactly norm-preserving, `expo` is
  skewed (nonzero third moment), `uniform` is bounded. Methods assuming Gaussian weights will
  show it in the per-strategy breakdown.
* Odd activations (`zgauss`, `tanh_rmsnorm`) have means below the benchmark's resolution, so
  those networks are flagged *uninformative* and excluded from the headline (predicting zero
  there is already at the floor). Return a sensible prediction for them, but do not spend effort
  there — the headline is decided on the other six activations.
* `tanh_rmsnorm` couples the neurons of a layer; a per-neuron model needs a model of the layer
  RMS.
* Budget utilisation ≤ 10% gives the full discount; there is no reward for using less. Spending
  part of the budget on *metered* sampling to correct an analytic estimate is legitimate.

## Deliverable

`estimator.py` in your working directory (plus any tables it loads in `setup()` from its own
folder), a short `NOTES.md` explaining the method, and your best test-split run directory.
