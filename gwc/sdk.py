"""What a submission sees: the estimator contract.

A submission is a Python file defining ``class Estimator(BaseEstimator)`` with a
``predict(net, budget)`` method. For each network it receives the full
:class:`gwc.netspec.Network` (width, depth, activation name, weight-strategy
name, the weight matrices, a seed) and a FLOP budget, and must return the
predicted per-layer, per-neuron activation MEANS as a ``(depth, width)`` array.

Rules (the grader assumes you follow them -- see README "Rules"):

* All numeric work inside ``predict`` must go through ``flopscope.numpy``
  (``import flopscope.numpy as fnp``) and :func:`gwc.sdk.activation`, so it is
  metered. Unmetered NumPy/other libraries in ``predict`` make the FLOP count
  (and therefore the score) meaningless.
* Do not read the grader's targets, key, or state files. Do not special-case
  network indices or names; solve the *type* (activation, strategy, shape).
* ``setup()`` runs once, off-budget, before any prediction (load tables etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import flopscope.numpy as fnp

from gwc.activations import NAMES as ACTIVATIONS, FORMULA as ACTIVATION_FORMULA, apply as activation
from gwc.netspec import DEPTHS, WIDTHS, Network, NetType
from gwc.weights import STRATEGIES

API_VERSION = "1.0"


@dataclass(frozen=True)
class SetupContext:
    api_version: str
    submission_dir: str
    seed: int
    n_ref: int                    # samples the budget buys, for every network
    activations: Sequence[str]    # all activation names that can appear
    strategies: Sequence[str]     # all weight strategies that can appear
    widths: Sequence[int]
    depths: Sequence[int]


class BaseEstimator:
    """Subclass this. Only ``predict`` is required."""

    def setup(self, ctx: SetupContext) -> None:  # optional, off-budget
        return None

    def predict(self, net: Network, budget: int) -> "fnp.ndarray":
        """Return the predicted activation means, shape (net.depth, net.width), float32.

        ``budget`` is the FLOP budget for this network (see gwc.budget). Work is
        metered by flopscope while this method runs; exceeding the budget fails
        the network (prediction zeroed, no compute discount).
        """
        raise NotImplementedError

    def teardown(self) -> None:  # optional
        return None


__all__ = [
    "API_VERSION", "BaseEstimator", "SetupContext", "Network", "NetType",
    "activation", "ACTIVATIONS", "ACTIVATION_FORMULA", "STRATEGIES", "WIDTHS", "DEPTHS", "fnp",
]
