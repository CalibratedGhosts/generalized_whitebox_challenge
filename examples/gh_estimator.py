"""Baseline: general Gaussian moment propagation with Gauss-Hermite quadrature.

Works for ANY activation without a closed form. Each neuron's pre-activation is
modelled as Gaussian N(mu, v); the first two moments of phi(N(mu, v)) are
computed by K-point Gauss-Hermite quadrature (metered), and propagated through
the next linear layer under a mean-field (independent-neurons) approximation:

    mu'_j = sum_i mu_i W_ij          v'_j = sum_i v_i W_ij^2

The independence assumption ignores correlations between neurons, so this is a
*starting point*, not the answer. For ``tanh_rmsnorm`` the layer RMS is
approximated by sqrt(mean_j(mu'_j^2 + v'_j)).

Cost per layer ~ K * width * cost(phi) + 2 * width^2  --  a tiny fraction of the
budget (multiplier sits at the 0.1 floor).
"""

from __future__ import annotations

import numpy as np
import flopscope.numpy as fnp

from gwc.sdk import BaseEstimator, Network, SetupContext, activation

K = 32


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
            if net.activation == "tanh_rmsnorm":
                rms = fnp.sqrt(fnp.mean(mu_pre * mu_pre + var_pre) + fnp.float32(1e-6))
                A = fnp.tanh(Z / rms)
            else:
                A = activation(net.activation, Z)
            m1 = A @ self._w                                             # E[phi]
            m2 = (A * A) @ self._w                                       # E[phi^2]
            mu = m1
            var = fnp.maximum(m2 - m1 * m1, fnp.float32(0.0))
            out.append(mu)
        return fnp.stack(out)
