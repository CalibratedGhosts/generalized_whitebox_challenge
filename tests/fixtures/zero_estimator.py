"""Baseline: predict all-zero means. Uses ~0 FLOPs (hits the 0.1 discount floor)."""

from __future__ import annotations

import flopscope.numpy as fnp

from gwc.sdk import BaseEstimator, Network


class Estimator(BaseEstimator):
    def predict(self, net: Network, budget: int):
        return fnp.zeros((net.depth, net.width), dtype=fnp.float32)
