"""The 8 activation functions of the Generalized WhiteBox Challenge.

Every network applies ONE of these element-wise (except ``tanh_rmsnorm``, which
couples the neurons of a layer through an RMS normalisation) after each linear
layer::

    a_0 = x ~ N(0, I_width)
    a_l = phi(a_{l-1} @ W_l)          l = 1..depth

The set was chosen (see docs/DESIGN.md) so that every activation is numerically
stable at every width/depth/weight-strategy in the challenge under a fixed
per-activation gain, while the moment maps m(mu, sigma) = E[phi(mu + sigma z)]
are pairwise non-redundant (no pair is affinely equivalent given mu).

Two implementations of the same function are provided:

* :func:`apply`    -- ``flopscope.numpy`` (METERED). Use this inside an estimator.
* :func:`apply_np` -- plain NumPy (unmetered). Used only for ground-truth
                     simulation by the grader. Estimators must not use it.
"""

from __future__ import annotations

import numpy as np
import flopscope.numpy as fnp

NAMES = ("relu", "relu2_sat", "sq_sat", "cos", "tanh_rmsnorm", "gabor", "rbump", "zgauss")

RMS_EPS = 1e-6

FORMULA = {
    "relu": "max(z, 0)",
    "relu2_sat": "r^2 / (1 + r^2),  r = max(z, 0)",
    "sq_sat": "z^2 / (1 + z^2)",
    "cos": "cos(z)",
    "tanh_rmsnorm": "tanh(z / sqrt(mean_j(z_j^2) + 1e-6))   [mean over the layer]",
    "gabor": "cos(2 z) * exp(-z^2 / 2)",
    "rbump": "max(z, 0) * exp(-z)",
    "zgauss": "z * exp(-z^2)",
}

CLASS = {
    "relu": "one-sided, linear tail",
    "relu2_sat": "one-sided, quadratic onset, saturating (stable ReLU^2)",
    "sq_sat": "even, quadratic onset, saturating (stable x^2)",
    "cos": "periodic, bounded",
    "tanh_rmsnorm": "odd, bounded, coupled across the layer",
    "gabor": "even, localised, oscillatory",
    "rbump": "one-sided, localised",
    "zgauss": "odd, localised",
}

# Per-activation gain g = 1 / sqrt(E[phi(z)^2]), z ~ N(0,1) (4M-sample estimate,
# rounded). Weight matrices are scaled by g / sqrt(width) so that, at every
# layer, the pre-activation variance stays ~1 -- the generalisation of He
# initialisation (which is exactly this rule for relu: g = sqrt(2)).
# tanh_rmsnorm is scale-invariant, so its gain is immaterial and set to 1.
GAIN = {
    "relu": 1.4145,
    "relu2_sat": 3.2571,
    "sq_sat": 2.3032,
    "cos": 1.3269,
    "tanh_rmsnorm": 1.0,
    "gabor": 1.7994,
    "rbump": 4.8450,
    "zgauss": 3.3432,
}

# Approximate flopscope float32 cost per element (documentation; the budget uses
# an exact measurement, see gwc.budget.act_flops_per_sample).
FLOPS_PER_ELEMENT_APPROX = {
    "relu": 1, "relu2_sat": 5, "sq_sat": 4, "cos": 16,
    "tanh_rmsnorm": 19, "gabor": 37, "rbump": 19, "zgauss": 19,
}


def _apply(name: str, z, xp):
    if name == "relu":
        return xp.maximum(z, 0.0)
    if name == "relu2_sat":
        r = xp.maximum(z, 0.0)
        r2 = r * r
        return r2 / (1.0 + r2)
    if name == "sq_sat":
        z2 = z * z
        return z2 / (1.0 + z2)
    if name == "cos":
        return xp.cos(z)
    if name == "tanh_rmsnorm":
        rms = xp.sqrt(xp.mean(z * z, axis=-1, keepdims=True) + RMS_EPS)
        return xp.tanh(z / rms)
    if name == "gabor":
        return xp.cos(2.0 * z) * xp.exp(-(z * z) / 2.0)
    if name == "rbump":
        # == max(z,0) * exp(-z), written so exp never overflows (r >= 0).
        r = xp.maximum(z, 0.0)
        return r * xp.exp(-r)
    if name == "zgauss":
        return z * xp.exp(-(z * z))
    raise KeyError(f"unknown activation {name!r}; valid: {NAMES}")


def apply(name: str, z):
    """METERED activation (flopscope.numpy). ``z`` is an (n, width) pre-activation."""
    return _apply(name, z, fnp)


def apply_np(name: str, z: np.ndarray) -> np.ndarray:
    """Unmetered NumPy activation. For ground-truth simulation only."""
    return _apply(name, z, np)


def gain(name: str) -> float:
    return GAIN[name]


def describe(name: str) -> str:
    return f"{name}: {FORMULA[name]}  [{CLASS[name]}]"
