"""The 4 weight-sampling strategies.

Every strategy produces a (width, width) float32 matrix whose entries have
zero mean and unit variance in *shape*, then scales it by ``gain / sqrt(width)``
(``gain`` from :mod:`gwc.activations`) so pre-activations keep ~unit variance
layer after layer. The strategies therefore differ in the *distribution* of the
entries (bounded / Gaussian / structured-orthogonal / right-skewed), not in
their scale::

    uniform : U(-1, 1) * sqrt(3)                       bounded, flat
    gauss   : N(0, 1)                                  Gaussian
    orth    : Haar-random orthogonal Q (rows and columns orthonormal), times gain
              (norm-preserving, so no 1/sqrt(width) factor)
    expo    : -1 + Exp(rate=1)                         floor at -1, mean 0, var 1,
              right-skewed (E[w^3] = 2)
"""

from __future__ import annotations

import numpy as np

STRATEGIES = ("uniform", "gauss", "orth", "expo")

DESCRIPTION = {
    "uniform": "entries U(-1,1)*sqrt(3) * gain/sqrt(width)  (bounded, flat)",
    "gauss": "entries N(0,1) * gain/sqrt(width)  (Gaussian)",
    "orth": "gain * Q, Q Haar-random orthogonal  (norm-preserving)",
    "expo": "entries (-1 + Exp(1)) * gain/sqrt(width)  (floor -1, mean 0, var 1, right-skewed)",
}


def sample_weight(strategy: str, width: int, gain: float, rng: np.random.Generator) -> np.ndarray:
    """One (width, width) float32 weight matrix for ``strategy``."""
    w = int(width)
    if strategy == "orth":
        a = rng.standard_normal((w, w))
        q, r = np.linalg.qr(a)
        q = q * np.sign(np.diag(r))  # Haar measure correction
        return (gain * q).astype(np.float32)
    if strategy == "uniform":
        m = rng.uniform(-1.0, 1.0, (w, w)) * np.sqrt(3.0)
    elif strategy == "gauss":
        m = rng.standard_normal((w, w))
    elif strategy == "expo":
        m = -1.0 + rng.exponential(1.0, (w, w))
    else:
        raise KeyError(f"unknown weight strategy {strategy!r}; valid: {STRATEGIES}")
    return (m * (gain / np.sqrt(w))).astype(np.float32)


def describe(strategy: str) -> str:
    return f"{strategy}: {DESCRIPTION[strategy]}"
