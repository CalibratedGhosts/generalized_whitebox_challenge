"""FLOP budget and score arithmetic.

The per-network FLOP budget is denominated in Monte-Carlo samples: it is the
number of FLOPs that ``N_REF`` metered forward samples through that network
would cost (activation-aware). So at every type the budget "buys" the same
number of samples, and a full-budget Monte-Carlo estimator lands at the same
relative accuracy everywhere -- that is the bar to beat.

    B(net) = N_REF * mc_flops_per_sample(width, depth, activation)

    mc_flops_per_sample = 16 w      standard_normal((n, w))  [flopscope, 16/elem]
                        +  w       array wrap               [1/elem]
                        + d * w(2w-1)   one matmul per layer
                        + d * A_phi(w)  activation per layer (measured, exact)
                        + 4 d w    float64 accumulation of the means (2/elem cast + 2/elem add)

which is whestbench's Phase-2 cost model with the ReLU term generalised.

Score arithmetic is reused from whestbench.budget so the compute discount is
exactly the challenge's: multiplier = max(0.1, C/B) (1.0 on failure).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import flopscope
import flopscope.numpy as fnp
from whestbench.budget import effective_compute, score_multiplier  # noqa: F401  (re-exported)

from gwc.activations import apply
from gwc.netspec import NetType

N_REF = 2**16                 # samples the budget buys, per network
GROUND_TRUTH_SAMPLES = 2**21  # samples used for the stored ground truth (32x N_REF)


@lru_cache(maxsize=None)
def act_flops_per_sample(activation: str, width: int) -> int:
    """Exact flopscope cost of applying ``activation`` to ONE (1, width) row.

    Measured on a (64, width) probe so per-row reductions (tanh_rmsnorm) are
    included; flopscope prices are deterministic so this is exact.
    """
    n = 64
    z = fnp.asarray(np.zeros((n, int(width)), dtype=np.float32))
    with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
        apply(activation, z)
    total = int(ctx.flops_used)
    if total % n:
        raise RuntimeError(f"activation cost not divisible by probe rows ({activation}, {width})")
    return total // n


def mc_flops_per_sample(width: int, depth: int, activation: str) -> int:
    w, d = int(width), int(depth)
    rng = 16 * w
    wrap = w
    matmul = d * w * (2 * w - 1)
    act = d * act_flops_per_sample(activation, w)
    accumulate = 4 * d * w
    return rng + wrap + matmul + act + accumulate


def flop_budget(ntype: NetType, n_ref: int = N_REF) -> int:
    return int(n_ref) * mc_flops_per_sample(ntype.width, ntype.depth, ntype.activation)


def mc_at_budget(avg_var_final: float, n_ref: int = N_REF) -> float:
    """Expected final-layer MSE of a full-budget Monte-Carlo estimator: sigma^2 / N_REF."""
    return float(avg_var_final) / float(n_ref)


def noise_floor(avg_var_final: float, n_gt: int = GROUND_TRUTH_SAMPLES) -> float:
    """Ground-truth noise (sigma^2 / G): measured MSE is biased upward by this."""
    return float(avg_var_final) / float(n_gt)
