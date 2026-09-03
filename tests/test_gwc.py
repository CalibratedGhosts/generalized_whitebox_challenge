"""Tests for the Generalized WhiteBox Challenge harness.

Run:  uv run pytest -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import flopscope
import flopscope.numpy as fnp

from gwc.activations import GAIN, NAMES, apply, apply_np
from gwc.budget import N_REF, act_flops_per_sample, flop_budget, mc_flops_per_sample, score_multiplier
from gwc.crypto import decrypt_targets, encrypt_targets
from gwc.groundtruth import simulate
from gwc.netspec import (DEPTHS, WIDTHS, NetType, Network, build_weights, probe_status, sample_networks,
                         sample_type, weights_sha256)
from gwc.scoring import aggregate, geo_mean, score_network
from gwc.weights import STRATEGIES, sample_weight

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------


def test_metered_and_unmetered_activations_agree():
    z = np.random.default_rng(0).standard_normal((64, 48)).astype(np.float32) * 2
    for a in NAMES:
        with flopscope.BudgetContext(flop_budget=10**12, quiet=True):
            m = np.asarray(apply(a, fnp.asarray(z)))
        assert m.dtype == np.float32, a
        np.testing.assert_allclose(m, apply_np(a, z), atol=2e-6, err_msg=a)


def test_gains_are_unit_second_moment_normalisers():
    # g = 1/sqrt(E[phi(z)^2]) for z ~ N(0,1)  (tanh_rmsnorm is scale-free -> 1)
    z = np.random.default_rng(1).standard_normal(2_000_000).astype(np.float32)
    for a in NAMES:
        if a.startswith("rmsnorm"):
            # layer-normalised: scale-free (the next layer renormalises); gain only needs to be sane
            assert 0.3 < GAIN[a] <= 1.0
            continue
        m2 = float(np.mean(apply_np(a, z).astype(np.float64) ** 2))
        assert abs(1.0 / np.sqrt(m2) - GAIN[a]) / GAIN[a] < 0.01, a


def test_activation_costs_are_positive_and_per_element():
    for a in NAMES:
        c16, c384 = act_flops_per_sample(a, 16), act_flops_per_sample(a, 384)
        assert c16 > 0 and c384 > 0
        # per-element cost is (nearly) width independent
        assert abs(c16 / 16 - c384 / 384) < 1.0, a


# ---------------------------------------------------------------------------
# weights / stability
# ---------------------------------------------------------------------------


def test_weight_strategies_have_expected_scale_and_shape():
    rng = np.random.default_rng(0)
    for s in STRATEGIES:
        W = sample_weight(s, 128, 1.0, rng)
        assert W.shape == (128, 128) and W.dtype == np.float32
        if s == "orth":
            np.testing.assert_allclose(W.T @ W, np.eye(128), atol=1e-4)
        else:
            assert abs(float(W.std()) - 1 / np.sqrt(128)) / (1 / np.sqrt(128)) < 0.05
            assert abs(float(W.mean())) < 0.01
        if s == "expo":
            assert float(W.min()) >= -1.0 / np.sqrt(128) - 1e-6  # floor at -1 (before scaling)


@pytest.mark.parametrize("a", NAMES)
def test_every_activation_is_stable_at_the_extremes(a):
    for s in STRATEGIES:
        for (w, d) in [(16, 4), (384, 24), (96, 12)]:
            t = NetType(w, d, a, s)
            st = probe_status(t, build_weights(t, 7))
            assert st in ("ok", "vanish"), (a, s, w, d, st)  # never NaN/huge; rare vanish is resampled
            if (w, d) != (16, 4):
                assert st == "ok" or (w <= 24), (a, s, w, d, st)


# ---------------------------------------------------------------------------
# dataset determinism
# ---------------------------------------------------------------------------


def test_sampling_is_deterministic_and_balanced():
    a = sample_networks(64, master_seed=123, keep_weights=False)
    b = sample_networks(64, master_seed=123, keep_weights=False)
    assert [(n.ntype, n.seed, n.weight_sha256) for n in a] == [(n.ntype, n.seed, n.weight_sha256) for n in b]
    assert all(n.ntype.width in WIDTHS and n.ntype.depth in DEPTHS for n in a)
    assert all(n.ntype.activation in NAMES and n.ntype.strategy in STRATEGIES for n in a)


def test_manifest_regeneration_matches_hashes():
    m = json.loads((ROOT / "data" / "manifest.json").read_text())
    assert m["n_networks"] == 512 and m["n_train"] == 256
    for e in m["networks"][:3] + m["networks"][-3:]:
        t = NetType(e["width"], e["depth"], e["activation"], e["strategy"])
        assert weights_sha256(build_weights(t, e["seed"])) == e["weight_sha256"]


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_budget_formula_and_sample_accounting():
    for t in [NetType(16, 4, "relu", "gauss"), NetType(384, 24, "gabor", "orth")]:
        per = mc_flops_per_sample(t.width, t.depth, t.activation)
        w, d = t.width, t.depth
        expected = 16 * w + w + d * w * (2 * w - 1) + d * act_flops_per_sample(t.activation, w) + 4 * d * w
        assert per == expected
        assert flop_budget(t) == N_REF * per


def test_metered_mc_sample_costs_what_the_budget_says():
    """One metered forward sample through a network costs mc_flops_per_sample (up to accumulation)."""
    t = NetType(32, 6, "cos", "gauss")
    Ws = [fnp.asarray(W) for W in build_weights(t, 3)]
    n = 128
    with flopscope.BudgetContext(flop_budget=10**12, quiet=True) as ctx:
        a = fnp.random.default_rng(0).standard_normal((n, t.width), dtype=fnp.float32)
        for W in Ws:
            a = apply(t.activation, a @ W)
    per = ctx.flops_used / n
    model = mc_flops_per_sample(t.width, t.depth, t.activation)
    assert abs(per - (model - t.width - 4 * t.depth * t.width)) < 1.0  # minus wrap + f64 accumulation terms


# ---------------------------------------------------------------------------
# ground truth + crypto
# ---------------------------------------------------------------------------


def test_simulate_is_deterministic_and_matches_raw_forward():
    t = NetType(24, 5, "relu", "expo")
    W = build_weights(t, 11)
    r1 = simulate(t, W, 4096, seed=5)
    r2 = simulate(t, W, 4096, seed=5)
    assert np.array_equal(r1["means"], r2["means"]) and r1["n_samples"] == 4096
    x = np.random.default_rng(np.random.SeedSequence(5)).standard_normal((4096, 24), dtype=np.float32)
    a = x
    for w in W:
        a = apply_np("relu", a @ w).astype(np.float32)
    np.testing.assert_allclose(r1["means"][-1], a.astype(np.float64).mean(0), atol=1e-5)


def test_targets_encrypt_decrypt_roundtrip():
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    t = NetType(16, 4, "cos", "uniform")
    r = simulate(t, build_weights(t, 1), 512, seed=1)
    back = decrypt_targets(encrypt_targets({3: r, 9: r}, key), key)
    assert set(back) == {3, 9} and np.array_equal(back[3]["means"], r["means"]) and back[9]["n_samples"] == 512
    with pytest.raises(Exception):
        decrypt_targets(encrypt_targets({3: r}, key), Fernet.generate_key())


# ---------------------------------------------------------------------------
# scoring semantics
# ---------------------------------------------------------------------------


def _net():
    return Network(0, "train", NetType(8, 3, "relu", "gauss"), [], 0, "toy")


def test_score_ratio_and_multiplier_semantics():
    means = np.full((3, 8), 0.5, np.float32)
    variances = np.full((3, 8), 0.2, np.float32)
    target = {"means": means, "vars": variances, "n_samples": 1_000_000}
    budget = 1000
    # perfect prediction, 5% of budget -> ratio 0, multiplier floor 0.1; adjusted floored at the resolution
    s = score_network(_net(), target, means.copy(), {"budget": budget, "flops_used": 50})
    assert s.mse_final == 0.0 and s.multiplier == 0.1 and s.beats_sampling
    assert s.adjusted_ratio == pytest.approx(s.resolution * 0.1) and s.resolution == pytest.approx(N_REF / 1_000_000)
    # error of 0.01 everywhere, full budget -> ratio = 1e-4 / (0.2/N_REF)
    pred = means + 0.01
    s = score_network(_net(), target, pred, {"budget": budget, "flops_used": budget})
    assert s.multiplier == pytest.approx(1.0)
    assert s.ratio_final == pytest.approx(1e-4 / (0.2 / N_REF), rel=1e-4)
    assert s.adjusted_ratio == pytest.approx(s.ratio_final)
    # failure: prediction zeroed and multiplier forced to 1 regardless of flops
    s = score_network(_net(), target, pred, {"budget": budget, "flops_used": 1, "failed": True, "failure": "timeout"})
    assert s.failed and s.multiplier == 1.0 and s.mse_final == pytest.approx(0.25)
    # bad output shape -> failure
    s = score_network(_net(), target, np.zeros((2, 8)), {"budget": budget, "flops_used": 1})
    assert s.failed and s.failure == "bad_output"
    assert score_multiplier(0, budget, failed=False) == 0.1


def test_geo_mean_and_aggregate():
    assert geo_mean([1, 100]) == pytest.approx(10.0)
    means = np.zeros((3, 8), np.float32)
    target = {"means": means, "vars": np.ones((3, 8), np.float32), "n_samples": 100}
    rows = [score_network(_net(), target, means, {"budget": 10, "flops_used": 10}),
            score_network(_net(), target, means + 1, {"budget": 10, "flops_used": 10})]
    agg = aggregate(rows)
    assert agg["n"] == 2 and agg["n_failed"] == 0
    assert agg["geo_adjusted_ratio"] == pytest.approx(geo_mean([r.adjusted_ratio for r in rows]))
    assert "relu" in agg["by_activation"] and agg["by_activation"]["relu"]["n"] == 2


# ---------------------------------------------------------------------------
# informative networks + resolution floor
# ---------------------------------------------------------------------------


def test_uninformative_targets_are_flagged_and_excluded_from_headline():
    from gwc.budget import GROUND_TRUTH_SAMPLES
    from gwc.scoring import resolution
    res = resolution()
    assert res == pytest.approx(N_REF / GROUND_TRUTH_SAMPLES)
    variances = np.full((3, 8), 0.4, np.float32)
    # target means far below ground-truth noise -> uninformative; zero prediction sits at the floor
    tiny = {"means": np.full((3, 8), 1e-5, np.float32), "vars": variances, "n_samples": GROUND_TRUTH_SAMPLES}
    s = score_network(_net(), tiny, np.zeros((3, 8)), {"budget": 100, "flops_used": 1})
    assert not s.informative and s.signal_ratio < 3 * res
    assert s.adjusted_ratio == pytest.approx(max(s.ratio_final, res) * 0.1)
    # resolvable target -> informative
    big = {"means": np.full((3, 8), 0.3, np.float32), "vars": variances, "n_samples": GROUND_TRUTH_SAMPLES}
    t = score_network(_net(), big, big["means"].copy(), {"budget": 100, "flops_used": 1})  # bit-exact prediction
    assert t.informative and t.ratio_final == 0.0 and t.adjusted_ratio == pytest.approx(res * 0.1)
    # headline uses informative networks only; the uninformative one is listed separately
    agg = aggregate([s, t])
    assert agg["n_informative"] == 1 and agg["n_uninformative"] == 1
    assert agg["geo_adjusted_ratio"] == pytest.approx(t.adjusted_ratio)
    assert agg["uninformative"][0]["index"] == s.index
