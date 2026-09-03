"""Estimator template. Copy this file, keep the class name `Estimator`, implement `predict`.

Interface
---------
predict(net, budget) -> array of shape (net.depth, net.width), float32

    net.width, net.depth         ints
    net.activation               one of gwc.sdk.ACTIVATIONS  (str)
    net.strategy                 one of gwc.sdk.STRATEGIES   (str)  -- how the weights were drawn
    net.weights                  list of net.depth numpy arrays, each (width, width) float32
    net.seed                     int (the stream the weights were drawn from)
    budget                       FLOP budget for this network (int); see README "Budget"

The network computes, for an input x ~ N(0, I):   a_0 = x,   a_l = phi(a_{l-1} @ W_l).
Return your estimate of E[a_l[i]] for every layer l and neuron i. Only the last
layer is scored; all layers are reported.

Everything numeric inside `predict` must go through `flopscope.numpy` (imported
below as `fnp`) and `activation(...)` so it is metered. Exceeding `budget`, a
timeout, an exception, or a wrong-shape / non-finite result fails the network.
`setup` runs once, before any prediction, off-budget.
"""

from __future__ import annotations

import flopscope.numpy as fnp

from gwc.sdk import BaseEstimator, Network, SetupContext, activation  # noqa: F401


class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        # ctx.activations, ctx.strategies, ctx.widths, ctx.depths, ctx.n_ref, ctx.seed, ctx.submission_dir
        pass

    def predict(self, net: Network, budget: int):
        # Example of metered use of the network's own nonlinearity:
        #   z = fnp.asarray(net.weights[0])          # wrap a weight matrix (metered)
        #   a = activation(net.activation, x @ z)    # apply the network's activation (metered)
        return fnp.zeros((net.depth, net.width), dtype=fnp.float32)
