"""The grader: runs a submission (in a subprocess) and scores it against the
encrypted ground truth of a split.

Splits
------
* ``train`` (networks 0-255): grade as often as you like.
* ``test``  (networks 256-511): the held-out set -- grading is allowed once
  every 4 hours (a local state file records submissions).
* ``smoke``: 8 small train networks (one per activation), for fast iteration.

The submission never sees the targets: the worker process only receives the
weights and the budget; this process decrypts the targets and scores.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from gwc.budget import N_REF
from gwc.crypto import decrypt_targets, load_key, secrets_dir
from gwc.netspec import N_TRAIN, Network, NetType, load_manifest
from gwc.scoring import NetScore, aggregate, format_report, rows_to_dicts, score_network

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
TARGETS = {"train": ROOT / "data" / "targets_train.enc", "test": ROOT / "data" / "targets_test.enc"}
RUNS_DIR = ROOT / "runs"
TEST_COOLDOWN_S = 4 * 3600
SMOKE_PER_ACTIVATION = 1


class CooldownError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# networks metadata (no weights needed in the parent)
# ---------------------------------------------------------------------------


def load_meta(split: Optional[str] = None, indices: Optional[Sequence[int]] = None) -> List[Network]:
    m = load_manifest(MANIFEST)
    want = set(indices) if indices is not None else None
    out = []
    for e in m["networks"]:
        if split is not None and e["split"] != split:
            continue
        if want is not None and e["index"] not in want:
            continue
        out.append(Network(e["index"], e["split"], NetType(e["width"], e["depth"], e["activation"], e["strategy"]),
                           [], e["seed"], e["name"], e["weight_sha256"]))
    return out


def smoke_indices() -> List[int]:
    """The cheapest train network of each activation (deterministic)."""
    nets = load_meta("train")
    best: Dict[str, Network] = {}
    for n in nets:
        cost = n.depth * n.width * n.width
        if n.activation not in best or cost < best[n.activation].depth * best[n.activation].width ** 2:
            best[n.activation] = n
    return sorted(n.index for n in best.values())


# ---------------------------------------------------------------------------
# test-split cooldown
# ---------------------------------------------------------------------------


def state_path() -> Path:
    return secrets_dir() / "test-submissions.json"


def _read_state() -> List[Dict]:
    p = state_path()
    return json.loads(p.read_text()) if p.exists() else []


def cooldown_remaining_s() -> float:
    st = _read_state()
    if not st:
        return 0.0
    last = max(float(r["time"]) for r in st)
    return max(0.0, TEST_COOLDOWN_S - (time.time() - last))


def _record_test(summary: Dict) -> None:
    st = _read_state()
    st.append({"time": time.time(), "utc": datetime.now(timezone.utc).isoformat(), **summary})
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=1))


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def load_targets(split: str, indices: Sequence[int], gt_dir: Optional[Path] = None) -> Dict[int, Dict]:
    if gt_dir is not None:  # raw (unencrypted) results, e.g. while building the dataset
        out = {}
        for i in indices:
            z = np.load(Path(gt_dir) / f"{i:03d}.npz")
            out[int(i)] = {"means": z["means"], "vars": z["vars"], "n_samples": int(z["n_samples"])}
        return out
    blob_path = TARGETS[split]
    if not blob_path.exists():
        raise FileNotFoundError(f"missing encrypted targets {blob_path}")
    all_t = decrypt_targets(blob_path.read_bytes(), load_key())
    missing = [i for i in indices if i not in all_t]
    if missing:
        raise KeyError(f"targets missing for networks {missing[:8]}...")
    return {int(i): all_t[int(i)] for i in indices}


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def run_worker(estimator: Path, class_name: Optional[str], indices: Sequence[int], run_dir: Path, *,
               predict_timeout_s: float, setup_timeout_s: float, seed: int, n_ref: int, quiet: bool) -> Dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    args = {
        "estimator": str(estimator), "class_name": class_name, "submission_dir": str(estimator.parent),
        "manifest": str(MANIFEST), "indices": [int(i) for i in indices], "seed": int(seed), "n_ref": int(n_ref),
        "predict_timeout_s": float(predict_timeout_s), "setup_timeout_s": float(setup_timeout_s),
        "out_npz": str(run_dir / "preds.npz"), "out_json": str(run_dir / "stats.json"),
    }
    (run_dir / "args.json").write_text(json.dumps(args, indent=1))
    overall = setup_timeout_s + len(indices) * (predict_timeout_s + 10.0) + 120.0
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(run_dir / "worker.log", "w") as logf:
        proc = subprocess.Popen([sys.executable, "-m", "gwc.worker", str(run_dir / "args.json")],
                                stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            proc.wait(timeout=overall)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
            raise RuntimeError(f"worker exceeded overall time limit ({overall:.0f}s); see {run_dir/'worker.log'}")
    if not (run_dir / "stats.json").exists():
        tail = (run_dir / "worker.log").read_text()[-2000:]
        raise RuntimeError(f"worker died (exit {proc.returncode}) without results. Log tail:\n{tail}")
    stats = {int(k): v for k, v in json.loads((run_dir / "stats.json").read_text()).items()}
    preds = {}
    if (run_dir / "preds.npz").exists():
        z = np.load(run_dir / "preds.npz")
        preds = {int(k[1:]): z[k] for k in z.files}
    return {"stats": stats, "preds": preds}


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------


def grade(estimator_path: "str | Path", split: str = "train", *, indices: Optional[Sequence[int]] = None,
          smoke: bool = False, predict_timeout_s: float = 300.0, setup_timeout_s: float = 120.0, seed: int = 0,
          n_ref: int = N_REF, force: bool = False, gt_dir: Optional[Path] = None, class_name: Optional[str] = None,
          quiet: bool = False, tag: str = "") -> Dict:
    estimator = Path(estimator_path).resolve()
    if estimator.is_dir():
        estimator = estimator / "estimator.py"
    if not estimator.is_file():
        raise FileNotFoundError(f"estimator file not found: {estimator}")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if smoke:
        split = "train"
        indices = smoke_indices()
    if split == "test" and not force:
        rem = cooldown_remaining_s()
        if rem > 0:
            raise CooldownError(f"test split was graded {TEST_COOLDOWN_S/3600:.0f}h-cooldown ago; "
                                f"{rem/60:.0f} min remaining. Use the train split meanwhile.")
    nets = load_meta(split, indices)
    if not nets:
        raise ValueError("no networks selected")
    idx = [n.index for n in nets]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = "smoke" if smoke else split
    run_dir = RUNS_DIR / f"{stamp}-{label}{('-' + tag) if tag else ''}"
    if not quiet:
        print(f"[gwc] grading {estimator.name} on {label} ({len(idx)} networks) -> {run_dir}", flush=True)
    t0 = time.time()
    w = run_worker(estimator, class_name, idx, run_dir, predict_timeout_s=predict_timeout_s,
                   setup_timeout_s=setup_timeout_s, seed=seed, n_ref=n_ref, quiet=quiet)
    targets = load_targets(split, idx, gt_dir=gt_dir)
    rows: List[NetScore] = [score_network(n, targets[n.index], w["preds"].get(n.index), w["stats"][n.index], n_ref=n_ref)
                            for n in nets]
    agg = aggregate(rows)
    result = {
        "estimator": str(estimator), "split": label, "n_networks": len(idx), "indices": idx,
        "n_ref": n_ref, "predict_timeout_s": predict_timeout_s, "wall_s": time.time() - t0,
        "run_dir": str(run_dir), "aggregate": agg, "rows": rows_to_dicts(rows),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=1, default=float))
    if split == "test" and not smoke:
        _record_test({"estimator": str(estimator), "geo_adjusted_ratio": agg["geo_adjusted_ratio"],
                      "geo_ratio": agg["geo_ratio"], "n_failed": agg["n_failed"], "run_dir": str(run_dir)})
    return result


def report(result: Dict) -> str:
    title = f"gwc  {Path(result['estimator']).name}  |  split={result['split']}  n={result['n_networks']}  wall={result['wall_s']:.0f}s"
    return format_report(result["aggregate"], title)
