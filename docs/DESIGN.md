# Design rationale

This document records *why* the challenge is defined the way it is, with the measurements
that drove each decision, so that the choices can be audited and revisited.

## 1. Goal

Push solvers toward **general** moment-estimation methods for neural networks — not methods
overfitted to one activation (ReLU), one initialisation (He/Gaussian), or one shape
(1024×16) — while keeping the ARC WhiteBox Challenge's measurement machinery (flopscope
metering, FLOP budgets, the `max(0.1, C/B)` compute discount).

## 2. Weight scaling: why a per-activation gain, not a bare `1/√width`

Requested: four raw weight distributions with `1/√width` scaling. Measured (width 384,
depth 24, 4096 probe inputs): with plain `1/√width`,

* `ReLU²`, `x²` with Gaussian or exponential weights → **NaN** (super-linear growth compounds);
  `ReLU`/`SiLU` × exponential → ~1e29 by depth 24 (overflows when squared in the MSE);
* `ReLU`, `SiLU`, `softsign` with uniform or orthogonal weights → activations **vanish to ~0**
  (one-sided activations attenuate; `1/√width` = LeCun scaling doesn't compensate);
  orthogonal × `1/√width` always decays by `1/√width` per layer;
* only the bounded/normalised activations (`cos`, `exp(−z²)`, `tanh(rms_norm)`) were stable.

The fix is the standard one generalised: scale each matrix by `g_φ/√width` with
`g_φ = 1/√E[φ(z)²]`, `z~N(0,1)` — the variance-preserving gain (He's `√2` is exactly this for
ReLU). Orthogonal matrices, being norm-preserving, get `g_φ` without the `1/√width`. The
exponential distribution is **centred** (`−1 + Exp(1)`: floor at −1, mean 0, var 1): with its
original mean +1 the mean amplifies coherently (`√width` per layer) and ReLU/SiLU nets reach
1e26. Centring keeps its distinctive right-skew (`E[w³] = 2`) and removes the blow-up.
Result: every strategy has zero-mean, unit-variance entries in shape; they differ in
boundedness, kurtosis, skew and structure.

## 3. Activations: stability forces "at most linear growth"

Under any fixed per-layer gain, the pre-activation variance obeys a recursion
`v → g² E[φ(√v z)²]`. For a degree-2 map (`x²`, `ReLU²`) this is `v → c v²`, whose only fixed
point has derivative 2 — **unstable for every gain**. Measured usable depth: `ReLU²` to depth
4–5, `x²` to depth 4 only; beyond that they vanish or explode for all strategies. Generally:

* bounded activations: always stable (contracting);
* linear tails (`relu`, `|z|`, leaky, Huber-like): neutral — fine to depth 24;
* super-linear anywhere near the operating scale: unstable. `pseudo-Huber` (`√(1+z²)−1`)
  explodes too, because at unit variance it is still in its quadratic core; `SiLU` drifts
  upward (its second-moment map is convex) and reached ~30× by depth 24.

So `x²` and `ReLU²` were replaced by the closest **stable** functions with the same symmetry and
onset: `sq_sat = z²/(1+z²)` (even, quadratic near 0, saturating) and
`relu2_sat = r²/(1+r²)` (one-sided, quadratic onset, saturating).

## 4. Activations: diversity as non-redundancy of moment maps

What matters for moment estimation is the map `m_φ(μ,σ) = E[φ(μ+σz)]`. Two activations are
redundant if one's map is an affine function of the other's *given the trivially known linear
term μ*. We fitted `m_j ≈ a·m_i + b·μ + c` on a grid `μ∈[−3,3], σ∈[0.25,2]` for a pool of 15
stable candidates. Exact or near-exact redundancies found: `relu ~ |z|` (R²=1.000, since
`relu = (z+|z|)/2`), `relu ~ elu` (0.997), `softsign ~ tanh` (0.994–0.998), `zgauss ~ softsign`
(0.990), `z²/(1+z²) ~ exp(−z²)` (0.987). An exhaustive search over 8-subsets that must contain
the requested `relu`, `cos`, `tanh(rms_norm)` and a stable ReLU²/x² analogue, minimising the
worst pairwise R², gave the final set (worst pair 0.984; everything ≥0.99 was excluded):

`relu, relu2_sat, sq_sat, cos, tanh_rmsnorm, gabor (cos 2z·e^{−z²/2}), rbump (r·e^{−r}), zgauss (z·e^{−z²})`

covering: one-sided linear, one-sided saturating, even saturating, periodic, odd-bounded-coupled,
even-localised-oscillatory, one-sided-localised, odd-localised. (With μ regressed out, ~0.98
is the floor for anything with "linear + bump" structure — even `cos ~ relu` scores 0.976 on
this grid — and the residual few percent is exactly the activation-specific structure an
MSE-to-1e-6 benchmark measures.) Full-grid verification (18×11×4×2 seeds per activation, 12,672
forward passes): zero NaN; six "vanish" corners at width 16–24 × depth 16–24, handled by the
validity resample.

## 5. Scoring: relative to sampling, geometric mean

Final-layer activation variances range from ~2e-4 (deep `cos`: the network collapses to a
near-deterministic output) to ~0.4 (`tanh_rmsnorm`); odd activations have means ~1e-3. Raw
MSEs are therefore incomparable across types and a raw average is dominated by whichever type
has the largest scale. Dividing each network's MSE by `σ²/N_REF` — the MSE that spending the
whole budget on Monte-Carlo attains — makes every network's score "how many times better/worse
than sampling", the same bar everywhere, and the geometric mean keeps any single type from
dominating. The compute discount is then applied exactly as in the challenge.

Two consequences worth knowing:

* **Odd activations are uninformative without biases.** With the real ground truth (2²¹
  samples) the final-layer means of every `tanh_rmsnorm` and `zgauss` network — for *all four*
  weight strategies, including the skewed exponential — are below the benchmark's resolution
  (`ratio_zero ≈ 1/32`): predicting zero is already at the floor. Left in a geometric mean, these
  free near-zero values dominated the headline (`cov_estimator` scored 0.74 overall while being
  3–17× worse than sampling on six activations). They are therefore flagged `informative=false`
  from the ground truth alone (`mean(target²)/(σ²/N_REF) ≤ 3·N_REF/G`), reported, and excluded
  from the headline; every ratio is also floored at the resolution `N_REF/G`. Train: 184/256
  informative, test: 199/256. **Recommended follow-up:** add a per-layer bias vector
  (`z = a@W + b`, `b ~ N(0, β²)`) to every network. That makes the odd activations — and all
  others — carry rich, resolvable per-neuron structure, is closer to real networks, and costs a
  ~75-minute re-precompute plus a `net.biases` field in the estimator API.
* Analytic Gaussian-propagation methods (mean-field or full-covariance) lose to sampling on most
  of this family — on the real ground truth `cov_estimator` is 6.4× worse than sampling overall
  (20× at width ≤ 64, 1.7× at width ≥ 256; it wins only on `gabor`), `gh_estimator` 34×. Beating
  sampling *everywhere* requires handling non-Gaussianity and correlations — the intended
  difficulty. Metered Monte-Carlo scores 0.971 (its geometric mean sits a few percent under 1
  because the per-network ratio is noisy at small width and the geometric mean of a noisy
  quantity is below its mean).

## 6. Ground-truth precision

`G = 2²¹ ≈ 2.1M` samples per network = 32× the budget's sample count, so the ground truth is
32× more precise than what a within-rules sampler can achieve and its noise (`σ²/G`) is 3% of
the sampling bar. The grader reports a bias-corrected ratio (`(mse − σ²/G)/(σ²/N_REF)`) and
flags networks at the noise floor, so methods more than ~30× better than sampling are still
measured meaningfully. Precompute cost: roughly 16 CPU-core-hours (parallel and resumable: `scripts/precompute.py`).

## 7. Differences from the ARC challenge

| | ARC WhiteBox (Phase 2) | gwc |
|---|---|---|
| activation | ReLU | 8, see §4 |
| weights | He `N(0, 2/w)` | 4 distributions, gain-scaled (§2) |
| shape | 1024×16 | 18 widths × 11 depths |
| budget | `2⁴¹` | `N_REF=2¹⁶` metered samples of each network (activation-aware cost) |
| score | mean of `mse·max(0.1,C/B)` | geometric mean of `(mse/(σ²/N_REF))·max(0.1,C/B)` |
| ground truth | live / 1e9 samples | precomputed 2²¹ samples, encrypted, bias-corrected reporting |
| isolation | subprocess / remote metering | subprocess; cooperative metering; honesty model |

The gwc ReLU networks use `g=√2` gain-scaled weights, i.e. He initialisation — the same
distribution as the challenge — but at different shapes and with the relative score, so gwc
numbers are not directly comparable to leaderboard MSEs; use the `whestgen` harness for exact
challenge parity.

## 8. Reproducibility

`data/manifest.json` fixes every network (type, weight seed, SHA-256 of the weights);
`gwc.netspec.load_networks` regenerates and verifies. Ground-truth seeds derive from a private
master seed; the targets are stored encrypted. Pinned: `numpy==2.2.6`, `flopscope==0.12.1`,
`whestbench==0.16.1`. Tests: `uv run pytest -q`.
