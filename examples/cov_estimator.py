"""Baseline: full-covariance Gaussian moment propagation, activation-agnostic.

The activation-agnostic analogue of the ARC WhiteBox reference solution. Each
layer's pre-activations are modelled as jointly Gaussian with mean ``mu`` and
covariance ``Sigma`` (the input is exactly N(0, I)). Through a linear layer:

    mu_pre = mu @ W            Sigma_pre = W^T Sigma W

Through the nonlinearity, per neuron, with K-point Gauss-Hermite quadrature on
the Gaussian marginal N(mu_i, Sigma_ii):

    m1_i = E[phi(z_i)]          m2_i = E[phi(z_i)^2]         (exact marginal moments)
    g_i  = E[phi(z_i)(z_i-mu_i)] / Sigma_ii  = E[phi'(z_i)]   (Stein's lemma; no derivative needed)

and the post-activation covariance is approximated by the gain (Bussgang-type)
rule for off-diagonals, exact on the diagonal:

    Cov(phi_i, phi_j) ~= g_i g_j Sigma_ij   (i != j),     Var(phi_i) = m2_i - m1_i^2

For the layer-normalised activations (``rmsnorm_sq``, ``rmsnorm_exp``) the
layer RMS is approximated by its Gaussian-moment expectation and treated as a
constant (see ``_nodes``).

Cost ~ 4 w^3 per layer (the two covariance matmuls) -- about 1% of the budget
at the largest shape, so the compute multiplier sits at the 0.1 floor. This is
a strong *general* starting point; it is not exact (the gain rule and the
Gaussian assumption after the first layer are approximations).
"""

from __future__ import annotations

import numpy as np
import flopscope.numpy as fnp

from gwc.sdk import BaseEstimator, Network, SetupContext, activation

K = 40


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
        x, w = np.polynomial.hermite_e.hermegauss(K)
        self._x = fnp.asarray(x.astype(np.float32))                # (K,)
        self._w = fnp.asarray((w / w.sum()).astype(np.float32))    # (K,), sums to 1

    def predict(self, net: Network, budget: int):
        w_ = net.width
        eye = fnp.eye(w_, dtype=fnp.float32)
        mu = fnp.zeros((w_,), dtype=fnp.float32)
        sig = eye
        out = []
        for W in net.weights:
            Wf = fnp.asarray(W)
            mu_pre = mu @ Wf
            sig_pre = Wf.T @ sig @ Wf
            v = fnp.maximum(fnp.sum(sig_pre * eye, axis=1), fnp.float32(1e-12))   # diagonal
            sd = fnp.sqrt(v)
            Zc = sd[:, None] * self._x[None, :]                                   # centred nodes (w, K)
            Z = mu_pre[:, None] + Zc
            A = _nodes(net.activation, Z, mu_pre, v)
            m1 = A @ self._w
            m2 = (A * A) @ self._w
            gain = ((A * Zc) @ self._w) / v                                        # E[phi'(z)] via Stein
            var_post = fnp.maximum(m2 - m1 * m1, fnp.float32(0.0))
            off = (gain[:, None] * gain[None, :]) * sig_pre
            sig = off * (fnp.float32(1.0) - eye) + var_post[:, None] * eye
            mu = m1
            out.append(mu)
        return fnp.stack(out)
