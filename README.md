# Generalized WhiteBox Challenge (gwc)

**Predict the mean activation of every neuron in a random MLP — from its weights alone,
without sampling it — within a FLOP budget — for *any* activation, weight distribution,
width and depth.**

This generalizes the [ARC WhiteBox Estimation Challenge](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)
(one activation, one shape) to a family of **18 widths × 11 depths × 8 activations × 4 weight
distributions**, so that a method can only score well by being genuinely general. It keeps the
challenge's machinery: `flopscope`-metered FLOPs, a per-network FLOP budget, and the
compute-discounted score.

* 512 networks, sampled independently (type first, then weights), split **256 train / 256 test**.
* Ground truth (Monte-Carlo, 2²¹ samples per network) is precomputed and stored **encrypted**.
* You submit a Python estimator; the grader runs it and returns rich per-type diagnostics.
* The **train** split can be graded without limit. The **test** split is graded **at most once
  every 4 hours** — it is the number that counts.

If you are an agent working on this: read [`AGENT_BRIEF.md`](AGENT_BRIEF.md) first.

---

## 1. The task

A network of type `(width w, depth d, activation φ, strategy s)` has `d` weight matrices
`W_1..W_d ∈ ℝ^{w×w}` (drawn by strategy `s`, scaled so pre-activations stay ~unit variance) and
computes, for an input `x ~ N(0, I_w)`:

```
a_0 = x
a_l = φ(a_{l-1} @ W_l)          l = 1..d          (row-vector convention: z = a @ W)
```

The **target** is the matrix of per-layer, per-neuron activation means
`M[l, i] = E_x[ a_l[i] ]`, shape `(d, w)`. Your estimator receives the network (weights,
activation name, strategy name, width, depth) and a FLOP budget, and returns its prediction of
`M`. The **final layer** (`M[d-1]`) is what is scored; all layers are reported.

### Activations

| name | φ(z) | class | metered cost / element |
|---|---|---|---|
| `relu` | `max(z,0)` | one-sided, linear tail | 1 |
| `relu2_sat` | `r²/(1+r²)`, `r=max(z,0)` | one-sided, quadratic onset, saturating | 5 |
| `sq_sat` | `z²/(1+z²)` | even, quadratic onset, saturating | 4 |
| `cos` | `cos(z)` | periodic, bounded | 16 |
| `tanh_rmsnorm` | `tanh(z / sqrt(mean_j z_j² + 1e-6))` | odd, bounded, **coupled across the layer** | ~19 |
| `gabor` | `cos(2z)·exp(−z²/2)` | even, localised, oscillatory | 37 |
| `rbump` | `r·exp(−r)`, `r=max(z,0)` | one-sided, localised | 19 |
| `zgauss` | `z·exp(−z²)` | odd, localised | 19 |

`tanh_rmsnorm` is the only non-element-wise one: the RMS is over the neurons of the layer
(for one input), so every neuron's output depends on the whole pre-activation vector.
Why these eight (and not `x²`, `ReLU²`, `SiLU`, ...): see [`docs/DESIGN.md`](docs/DESIGN.md) —
every activation must be numerically stable at depth 24 under a fixed gain (rules out anything
super-linear), and no two may be affinely equivalent as moment maps (rules out `|z|`, `elu`,
`silu`, `softsign`, ...).

### Weight strategies

All entries are scaled by `g_φ / sqrt(w)` where `g_φ = 1/sqrt(E[φ(z)²])` (the generalisation of
He initialisation), so the strategies differ in **shape**, not scale:

| name | entries | character |
|---|---|---|
| `uniform` | `U(−1,1)·√3` | bounded, flat |
| `gauss` | `N(0,1)` | Gaussian |
| `orth` | `g_φ · Q`, Q Haar-random orthogonal | norm-preserving, structured |
| `expo` | `−1 + Exp(rate 1)` | floor at −1, mean 0, right-skewed (E[w³]=2) |

Widths: `16 24 32 40 48 56 64 72 84 96 128 160 192 256 288 320 352 384`.
Depths: `4 5 6 7 8 10 12 14 16 20 24`. Type components are sampled independently and uniformly;
a draw is rejected (and redrawn) only if a forward probe is degenerate (NaN, or final RMS
outside `[1e-2, 1e2]`) — this affected ~0.1% of draws.

### FLOP budget

The budget is denominated in Monte-Carlo samples: it is the metered cost of `N_REF = 2¹⁶`
forward samples through *that* network,

```
B(net) = N_REF · [ 16w + w + d·w(2w−1) + d·A_φ(w) + 4dw ]
              rng    wrap   matmuls      activation   f64 accumulation
```

(`A_φ(w)` is the exact flopscope cost of the activation on one row). So at every type a
full-budget sampler reaches the same relative accuracy — that is the bar. Budgets range from
`1.7e8` (relu 16×4) to `4.9e11` (gabor 384×24).

## 2. Scoring

Raw MSEs are not comparable across activations (deep `cos` activations have variance ~2e-4,
`tanh_rmsnorm` ~0.4; odd activations have ~zero means). Every network is therefore scored
**relative to budget-matched sampling**:

```
ratio     = mse_final / (σ² / N_REF)          σ² = mean final-layer activation variance
                                             (σ²/N_REF = the MSE a full-budget MC estimator gets)
adjusted  = ratio · max(0.1, flops_used / B)  (multiplier forced to 1.0 on failure)
HEADLINE  = geometric mean of `adjusted` over the split's networks      (lower is better)
```

* `ratio < 1` beats sampling. `adjusted` applies the challenge's compute discount: use ≤10% of the
  budget and you get the maximal 10× discount; use it all and you get none.
* Geometric mean → scale-free; no single network type can dominate.
* **Failures** (exception, timeout, budget exhausted, wrong shape/non-finite output): the
  prediction is replaced by zeros and scored with multiplier 1.0 — bad, but bounded.
* **Bias-corrected ratio** `(mse − σ²/G)/(σ²/N_REF)`: the stored ground truth carries its own MC
  noise `σ²/G` (`G = 2²¹`, i.e. 1/32 of the sampling bar), which inflates every measured MSE by
  that amount. The corrected figure estimates your *true* error; `at_noise_floor` flags networks
  where you are within 2× of the ground-truth resolution.
* Also reported: worst-decile geometric mean (the generalisation tail), fraction of networks
  beating sampling, all-layers ratio, per-activation / per-strategy / per-width / per-depth
  breakdowns, budget utilisation, and the worst/best networks with error messages.

Reference points on the *train* split (see `examples/`): metered Monte-Carlo sits at
`adjusted ≈ 1` by construction (the calibration anchor); `zero` is ~10³; naive mean-field
Gaussian propagation and full-covariance Gaussian propagation beat sampling on wide/shallow
networks but are 10–500× *worse* than sampling on the narrowest ones. Being good everywhere
is the point.

## 3. Submitting

### Write an estimator

```python
# my_estimator.py
import flopscope.numpy as fnp                      # metered numpy -- use this for ALL math
from gwc.sdk import BaseEstimator, Network, SetupContext, activation

class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:    # optional, runs once, off-budget
        # ctx.activations, ctx.strategies, ctx.widths, ctx.depths, ctx.n_ref, ctx.seed
        ...
    def predict(self, net: Network, budget: int):  # metered, per network
        # net.width, net.depth, net.activation (str), net.strategy (str), net.seed
        # net.weights: list of (width, width) float32 numpy arrays  (wrap with fnp.asarray)
        # activation(net.activation, z): the metered activation function
        return fnp.zeros((net.depth, net.width), dtype=fnp.float32)   # shape (depth, width)
```

### Run it

```bash
cd <repo>
uv sync                                            # once
uv run gwc smoke  --estimator my_estimator.py      # 8 small train networks, seconds, unlimited
uv run gwc submit --estimator my_estimator.py      # full train split (256 networks), unlimited
uv run gwc submit --estimator my_estimator.py --split test    # held-out; once per 4 h
uv run gwc status                                  # cooldown + history
uv run gwc info                                    # dataset / types / budgets
```

Useful flags: `--indices 3,7,9` (train only: specific networks), `--timeout S` (per-network
wall cap, default 300 s), `--json` (full machine-readable result), `--tag NAME`. Every run is
saved under `runs/<timestamp>-<split>/` with `result.json` (all per-network rows), `worker.log`
(your estimator's stderr/tracebacks) and `preds.npz`.

Python API: `from gwc.grader import grade, report; r = grade("my_estimator.py", "train"); print(report(r))`.

### Rules (the grader assumes you follow them)

1. **All numeric work in `predict` goes through `flopscope.numpy` (`fnp`) and `gwc.sdk.activation`.**
   Unmetered NumPy/SciPy/torch inside `predict` makes the FLOP count — and thus the score —
   meaningless (the reported multiplier would be wrong). `setup()` is off-budget and unmetered.
2. Do not read `data/targets_*.enc`, the key in `$GWC_SECRETS_DIR` (default `~/.gwc`), the
   cooldown state, or `data/cache/`. Do not tamper with the cooldown.
3. Do not special-case network indices, names or seeds. Solve the *type*. (Note `net.seed` is
   the stream the weights were drawn from — if you sample, derive an independent stream from it,
   e.g. `fnp.random.default_rng(fnp.random.SeedSequence([net.seed, 1]))`.)
4. Return a finite `(depth, width)` array for every network.

## 4. Splits and the test cooldown

* `train` = networks 0–255. Grade it as often as you like; iterate here.
* `test` = networks 256–511, same distribution, never used for development. The grader refuses
  a test run within **4 hours** of the previous one (state in `$GWC_SECRETS_DIR/test-submissions.json`).
  `gwc status` shows the remaining time and your test history. Treat test results as the
  score of record; a large train–test gap means you have over-fitted the train networks.

## 5. Baselines (`examples/`)

| file | idea | typical behaviour |
|---|---|---|
| `zero_estimator.py` | predict 0 | terrible, except on odd activations (whose true means are ~0) |
| `mc_estimator.py` | honest metered Monte-Carlo, 90% of budget, float64 accumulation | `adjusted ≈ 1.0` everywhere — the calibration anchor |
| `gh_estimator.py` | mean-field Gaussian propagation, Gauss–Hermite moments | cheap (0.1 floor); ignores correlations → poor at small width |
| `cov_estimator.py` | full-covariance Gaussian propagation, Stein-lemma gains, GH moments | the general analogue of the ARC reference; good wide/shallow, poor narrow/deep |

## 6. Layout

```
gwc/activations.py   the 8 activations (metered `apply`, unmetered `apply_np`), gains
gwc/weights.py       the 4 weight strategies
gwc/netspec.py       types, deterministic sampler, validity probe, manifest (+hash-verified regeneration)
gwc/budget.py        cost model, per-network budget, MC reference, score multiplier (from whestbench)
gwc/groundtruth.py   Monte-Carlo ground truth (parallel, resumable)
gwc/crypto.py        Fernet encryption of targets (key never in the repo)
gwc/worker.py        subprocess that runs a submission under flopscope budgets + timeouts
gwc/scoring.py       per-network scores, geometric-mean aggregation, breakdowns, report
gwc/grader.py        orchestration, splits, smoke set, 4-hour test cooldown
gwc/cli.py           `gwc smoke | submit | status | info | build-targets`
data/manifest.json   the 512 network types + weight seeds + weight hashes (weights regenerate from these)
data/targets_*.enc   encrypted ground truth per split
examples/            baselines        tests/  pytest suite        docs/DESIGN.md  design rationale
```

Weights are regenerated from the manifest seeds at load time and verified against SHA-256
hashes, so a different NumPy/BLAS that reproduces different bits fails loudly rather than
silently changing the task (pinned: `numpy==2.2.6`, `flopscope==0.12.1`, `whestbench==0.16.1`).

## 7. Operator notes

* Ground truth: `uv run python scripts/precompute.py` (parallel, resumable), then
  `uv run gwc build-targets` to encrypt `data/cache/gt/*.npz` into `data/targets_{train,test}.enc`.
  The Fernet key is created at `$GWC_SECRETS_DIR/gwc-fernet.key` (default `~/.gwc`) and must be
  present on any machine that grades. **Never commit the key, the raw `data/cache/`, or `runs/`.**
* `gwc submit --split test --force` bypasses the cooldown (operator use only).
* Trust model: the targets are encrypted at rest and the submission runs in a separate process
  that never receives them. This is an honesty barrier and an over-fitting rate-limiter, not a
  defence against an adversary who controls the machine (who could read the key). FLOP metering
  is cooperative (rule 1). If you need adversarial guarantees, grade on a machine the submission
  cannot access.
