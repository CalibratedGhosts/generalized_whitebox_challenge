"""Baseline: general Gaussian moment propagation with Gauss-Hermite quadrature.

Works for ANY activation without a closed form. Each neuron's pre-activation is
modelled as Gaussian N(mu, v); the first two moments of phi(N(mu, v)) are
computed by K-point Gauss-Hermite quadrature (metered), and propagated through
the next linear layer under a mean-field (independent-neurons) approximation:

    mu'_j = sum_i mu_i W_ij          v'_j = sum_i v_i W_ij^2

The independence assumption ignores correlations between neurons, so this is a
*starting point*, not the answer. For the layer-normalised activations
(``rmsnorm_sq``, ``rmsnorm_exp``) the layer RMS is approximated by its
Gaussian-moment expectation and treated as a constant (see ``_nodes``).

Cost per layer ~ K * width * cost(phi) + 2 * width^2  --  a tiny fraction of the
budget (multiplier sits at the 0.1 floor).
"""

from __future__ import annotations

import numpy as np
import flopscope.numpy as fnp

from gwc.sdk import BaseEstimator, Network, SetupContext, activation

K = 32


def _nodes(name, Z, mu_pre, v):
    """phi at the quadrature nodes. Layer-normalised activations use the *expected*
    layer RMS (from the Gaussian moments) as a constant -- an approximation."""
    if name == "rmsnorm_sq":
        r2 = fnp.mean(mu_pre * mu_pre + v) + fnp.float32(1e-6)                        # E[mean_j z_j^2]
        return (Z * Z) / r2
    if name == "rmsnorm_exp":
        e = fnp.exp(fnp.minimum(Z, 60.0))
        r2 = fnp.mean(fnp.exp(fnp.minimum(2.0 * mu_pre + 2.0 * v, 120.0))) + fnp.float32(1e-6)  # E[e^{2z}]
        return e / fnp.sqrt(r2)
    return activation(name, Z)


class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        x, w = np.polynomial.hermite_e.hermegauss(K)      # probabilists' Hermite: weight exp(-x^2/2)
        self._x = fnp.asarray(x.astype(np.float32))         # (K,)
        self._w = fnp.asarray((w / w.sum()).astype(np.float32))  # normalised -> expectation under N(0,1)

    def predict(self, net: Network, budget: int):
        w_, d = net.width, net.depth
        mu = fnp.zeros((w_,), dtype=fnp.float32)
        var = fnp.ones((w_,), dtype=fnp.float32)
        out = []
        for W in net.weights:
            Wf = fnp.asarray(W)
            mu_pre = mu @ Wf
            var_pre = var @ (Wf * Wf)
            sd = fnp.sqrt(var_pre + fnp.float32(1e-12))
            Z = mu_pre[:, None] + sd[:, None] * self._x[None, :]        # (width, K) quadrature nodes
            A = _nodes(net.activation, Z, mu_pre, var_pre)
            m1 = A @ self._w                                             # E[phi]
            m2 = (A * A) @ self._w                                       # E[phi^2]
            mu = m1
            var = fnp.maximum(m2 - m1 * m1, fnp.float32(0.0))
            out.append(mu)
        return fnp.stack(out)
