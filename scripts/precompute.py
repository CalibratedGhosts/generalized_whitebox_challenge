"""Operator script: precompute the Monte-Carlo ground truth for all networks.

    uv run python scripts/precompute.py [--workers 8] [--samples 2097152]

Parallel (one single-threaded-BLAS process per worker) and resumable: networks
whose result file already exists under data/cache/gt/ are skipped. The
ground-truth master seed is created once under $GWC_SECRETS_DIR and never
committed. Afterwards run ``uv run gwc build-targets`` to encrypt the results.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"  # must precede numpy import: single-threaded BLAS per worker

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    from gwc.budget import GROUND_TRUTH_SAMPLES
    from gwc.crypto import secrets_dir
    from gwc.groundtruth import precompute
    from gwc.netspec import NetType, Network

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--samples", type=int, default=GROUND_TRUTH_SAMPLES)
    ap.add_argument("--out", default=str(ROOT / "data" / "cache" / "gt"))
    a = ap.parse_args()

    seed_path = secrets_dir() / "gwc-gt-seed.txt"
    if not seed_path.exists():
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(str(int.from_bytes(os.urandom(8), "big")))
        os.chmod(seed_path, 0o600)
    gt_seed = int(seed_path.read_text().strip())

    m = json.loads((ROOT / "data" / "manifest.json").read_text())
    nets = [Network(e["index"], e["split"], NetType(e["width"], e["depth"], e["activation"], e["strategy"]),
                    [], e["seed"], e["name"], e["weight_sha256"]) for e in m["networks"]]
    t0 = time.time()
    precompute(nets, Path(a.out), gt_seed, a.samples, workers=a.workers, verbose=True)
    print(f"PRECOMPUTE COMPLETE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
