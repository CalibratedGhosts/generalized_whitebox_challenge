"""End-to-end: worker + grader on a tiny locally-simulated ground truth (no encrypted targets needed)."""
from pathlib import Path

import numpy as np
import pytest

from gwc import grader
from gwc.groundtruth import simulate
from gwc.netspec import build_weights

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tiny_gt(tmp_path_factory):
    d = tmp_path_factory.mktemp("gt")
    nets = sorted(grader.load_meta("train"), key=lambda n: n.depth * n.width ** 2)
    chosen, seen = [], set()
    for n in nets:  # two cheapest train networks with different activations
        if n.activation not in seen:
            chosen.append(n); seen.add(n.activation)
        if len(chosen) == 2:
            break
    for n in chosen:
        r = simulate(n.ntype, build_weights(n.ntype, n.seed), 2**15, seed=1)
        np.savez(d / f"{n.index:03d}.npz", index=np.int64(n.index), **r)
    return d, [n.index for n in chosen]


def test_grader_end_to_end_zero_and_mc(tiny_gt):
    d, idx = tiny_gt
    z = grader.grade(ROOT / "tests/fixtures/zero_estimator.py", "train", indices=idx, gt_dir=d, quiet=True, predict_timeout_s=120)
    assert z["n_networks"] == 2 and z["aggregate"]["n_failed"] == 0
    assert all(r["flops_used"] == 0 and r["multiplier"] == 0.1 for r in z["rows"])
    m = grader.grade(ROOT / "tests/fixtures/mc_estimator.py", "train", indices=idx, gt_dir=d, quiet=True, predict_timeout_s=300)
    assert m["aggregate"]["n_failed"] == 0
    for r in m["rows"]:
        assert 0.80 < r["utilization"] < 0.95
        # G=2^15 -> ground-truth floor = N_REF/G = 2 in ratio units; MC ~ 1/0.9 + 2 ~ 3.1 (wide noise band)
        assert 1.0 < r["ratio_final"] < 8.0, r
    assert Path(m["run_dir"], "result.json").exists() and Path(m["run_dir"], "worker.log").exists()


def test_grader_reports_failures_and_bad_output(tiny_gt, tmp_path):
    d, idx = tiny_gt
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from gwc.sdk import BaseEstimator\nimport flopscope.numpy as fnp\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, net, budget):\n"
        f"        if net.index == {idx[0]}: raise RuntimeError('boom')\n"
        "        return fnp.zeros((net.depth + 1, net.width), dtype=fnp.float32)\n")
    r = grader.grade(bad, "train", indices=idx, gt_dir=d, quiet=True, predict_timeout_s=60)
    rows = {x["index"]: x for x in r["rows"]}
    assert rows[idx[0]]["failure"] == "error" and "boom" in rows[idx[0]]["error"]
    assert rows[idx[1]]["failure"] == "bad_output"
    assert r["aggregate"]["n_failed"] == 2 and all(x["multiplier"] == 1.0 for x in r["rows"])


def test_smoke_indices_one_per_activation():
    idx = grader.smoke_indices()
    nets = grader.load_meta("train", idx)
    assert len(idx) == 8 and len({n.activation for n in nets}) == 8 and all(n.split == "train" for n in nets)
