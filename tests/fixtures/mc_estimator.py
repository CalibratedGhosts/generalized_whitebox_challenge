"""Baseline: honest, fully-metered Monte-Carlo sampling within the budget.

Spends ~90% of the FLOP budget drawing standard-normal inputs and running the
network forward (all through flopscope), averaging activations per layer. By
construction its MSE is ~ sigma^2 / (0.9 N_REF), i.e. ratio ~ 1.1, and its
compute multiplier ~ 0.9, so its adjusted ratio is ~ 1.0. That is what the
whole benchmark is normalised to -- a calibration baseline.
"""

from __future__ import annotations

import flopscope.numpy as fnp

from gwc.budget import mc_flops_per_sample
from gwc.sdk import BaseEstimator, Network, activation

BUDGET_FRACTION = 0.90


class Estimator(BaseEstimator):
    def predict(self, net: Network, budget: int):
        w, d = net.width, net.depth
        per_sample = mc_flops_per_sample(w, d, net.activation)
        n_total = max(1, int(BUDGET_FRACTION * budget) // per_sample)
        chunk = max(64, min(4096, (1 << 20) // w))
        # NOTE: net.seed is the stream the WEIGHTS were drawn from. Re-using it
        # for the sampling inputs correlates x with W and biases the estimate.
        # Always derive an independent stream from it.
        rng = fnp.random.default_rng(fnp.random.SeedSequence([net.seed, 0x6D63]))
        weights = [fnp.asarray(W) for W in net.weights]
        # Accumulate in float64 (billed 2x, included in the budget's cost model):
        # float32 running sums over ~65k samples lose ~1e-3 absolute precision,
        # which is comparable to the MC noise itself and would inflate the MSE.
        sums = [fnp.zeros((w,), dtype=fnp.float64) for _ in range(d)]
        done = 0
        while done < n_total:
            n = min(chunk, n_total - done)
            a = rng.standard_normal((n, w), dtype=fnp.float32)
            for l, W in enumerate(weights):
                a = activation(net.activation, a @ W)
                sums[l] = sums[l] + fnp.sum(a.astype(fnp.float64), axis=0)
            done += n
        return fnp.stack([(s / fnp.float64(done)).astype(fnp.float32) for s in sums])
