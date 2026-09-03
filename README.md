# Generalized WhiteBox Challenge — grading service

Predict the mean activation of every neuron of a random MLP from its weights, within a FLOP
budget, across many widths, depths, activation functions and weight distributions. This page is
the complete description of the task and of the grading system.

## 1. The task

A network has `depth` weight matrices `W_1..W_depth`, each `(width, width)` float32, and one
activation `φ`. For an input `x ~ N(0, I_width)` it computes (row-vector convention):

```
a_0 = x
a_l = φ(a_{l-1} @ W_l)          l = 1 .. depth
```

The **target** is `M[l, i] = E_x[ a_l[i] ]`, the expected activation of every neuron in every
layer, shape `(depth, width)`. Your estimator receives the network and a FLOP budget and returns
its estimate of `M`. The last layer `M[depth-1]` is what is scored; all layers are reported.

### Activations

| name | φ(z) | notes |
|---|---|---|
| `relu` | `max(z, 0)` | |
| `relu2_sat` | `r² / (1 + r²)`, `r = max(z, 0)` | |
| `sq_sat` | `z² / (1 + z²)` | |
| `cos` | `cos(z)` | |
| `gabor` | `cos(2z) · exp(−z²/2)` | |
| `rbump` | `r · exp(−r)`, `r = max(z, 0)` | |
| `rmsnorm_sq` | `r²`, `r = z / sqrt(mean_j z_j² + 1e-6)` | mean over the layer, per input |
| `rmsnorm_exp` | `e / sqrt(mean_j e_j² + 1e-6)`, `e = exp(min(z, 60))` | mean over the layer, per input |

The first six act element-wise. The last two normalise over the neurons of the layer for each
input, so a neuron's output depends on the whole pre-activation vector.

### Weight distributions (`net.strategy`)

Each matrix is drawn i.i.d. and scaled by `g_φ / sqrt(width)`, where `g_φ = 1/sqrt(E[φ(z)²])`
for `z ~ N(0,1)` (`g_φ` is given in `gwc.activations.GAIN`):

| name | entries before scaling |
|---|---|
| `uniform` | `U(−1, 1) · sqrt(3)` |
| `gauss` | `N(0, 1)` |
| `orth` | a Haar-random orthogonal matrix, scaled by `g_φ` only (no `1/sqrt(width)`) |
| `expo` | `−1 + Exp(rate 1)` (support `[−1, ∞)`, mean 0, variance 1) |

### Dataset

512 networks. Each is drawn by first sampling its type — width, depth, activation, weight
distribution, each uniformly and independently — and then its weights. Widths
`16 24 32 40 48 56 64 72 84 96 128 160 192 256 288 320 352 384`; depths
`4 5 6 7 8 10 12 14 16 20 24`. Networks `0–255` are the **train** split, `256–511` the **test**
split. `data/manifest.json` fixes every network (type, weight seed, weight hash); weights are
regenerated from the seed and hash-checked when loaded. Ground truth is Monte-Carlo with
`2²¹` samples per network, stored encrypted; the grader decrypts it in memory.

### Budget

`budget(net) = 2¹⁶ · c(net)`, where `c` is the metered cost of pushing one sample through the
network:

```
c = 16w + w + d·w·(2w − 1) + d·A_φ(w) + 4dw       (w = width, d = depth)
```

`A_φ(w)` is the exact metered cost of the activation on one row. `gwc info --index N` prints a
network's budget.

## 2. Writing an estimator

```python
import flopscope.numpy as fnp                       # metered numpy: use it for ALL arithmetic in predict
from gwc.sdk import BaseEstimator, Network, SetupContext, activation

class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:     # optional; runs once, off-budget
        ...                                         # ctx.activations, ctx.strategies, ctx.widths, ctx.depths, ctx.n_ref
    def predict(self, net: Network, budget: int):   # metered; one call per network
        # net.width, net.depth, net.activation, net.strategy, net.seed
        # net.weights: list of (width, width) float32 numpy arrays   -> wrap with fnp.asarray(W)
        # activation(net.activation, z): the network's activation, metered
        return fnp.zeros((net.depth, net.width), dtype=fnp.float32)   # shape (depth, width), float32
```

A runnable copy of this is `examples/estimator_template.py`. The file must define a class named
`Estimator` (or pass `--class-name`).

## 3. Commands

```bash
uv sync                                                    # once
uv run gwc info                                            # dataset, activations, budgets
uv run gwc smoke  --estimator my_estimator.py              # 8 small train networks (seconds); unlimited
uv run gwc submit --estimator my_estimator.py              # the 256 train networks; unlimited
uv run gwc submit --estimator my_estimator.py --split test # the 256 test networks; once every 4 hours
uv run gwc status                                          # test cooldown and history
```

Options: `--indices 3,7,9` (train only: chosen networks), `--timeout S` (per-network wall-clock
cap, default 300 s), `--tag NAME`, `--json` (full result on stdout).

Every run is written to `runs/<timestamp>-<split>/`: `result.json` (aggregate + one row per
network with every number below), `worker.log` (your estimator's stderr and tracebacks),
`preds.npz` (your predictions). Python: `from gwc.grader import grade, report`.

## 4. Scoring

For each network, with `σ²` the mean variance of its last-layer activations:

```
ratio     = mse_last_layer / (σ² / 2¹⁶)           1.0 = the MSE of spending the whole budget on
                                                   plain Monte-Carlo sampling of this network
adjusted  = max(ratio, 1/32) · max(0.1, flops_used / budget)
HEADLINE  = geometric mean of `adjusted` over the scored networks of the split   (lower is better)
```

* `flops_used` is what flopscope metered inside `predict`. The multiplier is forced to `1.0` for a
  failed network (exception, timeout, budget exceeded, wrong shape or non-finite output), whose
  prediction is replaced by zeros.
* `1/32` is the resolution of the stored ground truth (its own Monte-Carlo noise); nothing can be
  measured below it.
* A network is **excluded from the headline** (but reported) if its target cannot separate
  methods: *uninformative* (target energy below the ground-truth resolution) or *degenerate*
  (`σ² < 1e-6`, near-deterministic output). The current dataset has 0 uninformative and 1
  degenerate network (`#305`, test split).

The report also gives: raw and bias-corrected geometric-mean ratios, the all-layers ratio, the
worst-decile geometric mean, the fraction of networks with `ratio < 1`, budget utilisation,
breakdowns by activation / weight distribution / width / depth, and the best and worst networks
with error messages. Per-network rows are in `result.json`.

## 5. Rules

1. All arithmetic inside `predict` goes through `flopscope.numpy` (`fnp`) and `gwc.sdk.activation`.
   Unmetered NumPy/SciPy/torch inside `predict` is not allowed. `setup()` is off-budget.
2. Do not read `data/targets_*.enc`, the key or state under `$GWC_SECRETS_DIR` (default `~/.gwc`),
   or `runs/` of other estimators; do not alter the cooldown state.
3. Do not special-case network indices, names or seeds. The same code must handle every type.
4. Return a finite float32 array of shape `(depth, width)` for every network.

## 6. Operator notes

Ground truth: `uv run python scripts/precompute.py` (parallel, resumable, writes
`data/cache/gt/`), then `uv run gwc build-targets` to encrypt into `data/targets_*.enc`. The
Fernet key is created at `$GWC_SECRETS_DIR/gwc-fernet.key`; the test-split cooldown state is
`$GWC_SECRETS_DIR/test-submissions.json`. Never commit the key, `data/cache/`, or `runs/`.
`gwc submit --split test --force` bypasses the cooldown. Tests: `uv run pytest -q`.
Pinned: `numpy==2.2.6`, `flopscope==0.12.1`, `whestbench==0.16.1`.
