"""Monte-Carlo ground truth: per-layer, per-neuron mean and variance of activations.

Uses plain (unmetered, float32/BLAS) NumPy -- this is the grader's reference,
not a submission. Means and second moments are accumulated in float64.

``precompute`` runs one process per worker with single-threaded BLAS (small
matrices don't benefit from BLAS threads, and 8 single-threaded workers beat
one 8-threaded process). It is resumable: networks with an existing result
file are skipped.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from gwc.activations import apply_np
from gwc.netspec import Network, NetType, build_weights


def simulate(ntype: NetType, weights: Sequence[np.ndarray], n_samples: int, seed: int,
             chunk: int = 8192) -> Dict[str, np.ndarray]:
    """Return {'means': (d,w) f32, 'vars': (d,w) f32, 'n_samples': int}."""
    w, d = ntype.width, ntype.depth
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    s1 = np.zeros((d, w), dtype=np.float64)
    s2 = np.zeros((d, w), dtype=np.float64)
    done = 0
    while done < n_samples:
        n = min(chunk, n_samples - done)
        a = rng.standard_normal((n, w), dtype=np.float32)
        for l, W in enumerate(weights):
            a = apply_np(ntype.activation, a @ W).astype(np.float32, copy=False)
            a64 = a.astype(np.float64)
            s1[l] += a64.sum(axis=0)
            s2[l] += (a64 * a64).sum(axis=0)
        done += n
    means = s1 / done
    variances = np.maximum(s2 / done - means * means, 0.0)
    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(variances))):
        raise FloatingPointError(f"non-finite ground truth for {ntype.key()}")
    return {"means": means.astype(np.float32), "vars": variances.astype(np.float32),
            "n_samples": np.int64(done)}


def gt_seed_for(gt_master_seed: int, index: int) -> int:
    ss = np.random.SeedSequence([int(gt_master_seed), int(index)])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def _job(args):
    index, width, depth, activation, strategy, wseed, n_samples, gt_seed, out_path = args
    ntype = NetType(width, depth, activation, strategy)
    weights = build_weights(ntype, wseed)
    t0 = time.time()
    res = simulate(ntype, weights, n_samples, gt_seed)
    np.savez(out_path, index=np.int64(index), **res)
    return index, time.time() - t0


def precompute(nets: Sequence[Network], out_dir: Path, gt_master_seed: int, n_samples: int,
               workers: int = 8, verbose: bool = True) -> List[Path]:
    """Compute (resumably, in parallel) the ground truth of every network in ``nets``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    paths = []
    for n in nets:
        p = out_dir / f"{n.index:03d}.npz"
        paths.append(p)
        if p.exists():
            continue
        jobs.append((n.index, n.width, n.depth, n.activation, n.strategy, n.seed, int(n_samples),
                     gt_seed_for(gt_master_seed, n.index), str(p)))
    # Biggest networks first for better load balance.
    jobs.sort(key=lambda j: -(j[2] * j[1] * j[1]))
    if verbose:
        print(f"precompute: {len(jobs)} networks to do ({len(nets) - len(jobs)} cached), "
              f"G={n_samples:,} samples each, {workers} workers", flush=True)
    if not jobs:
        return paths
    env = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update(env)  # inherited by spawned workers -> single-threaded BLAS each
    t0 = time.time()
    done = 0
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            idx, dt = f.result()
            done += 1
            if verbose and (done % 16 == 0 or done == len(jobs)):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)} done, {el/60:.1f} min elapsed, "
                      f"eta {(el/done)*(len(jobs)-done)/60:.1f} min (last: #{idx} {dt:.1f}s)", flush=True)
    return paths


def load_ground_truth(out_dir: Path, indices: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
    out = {}
    for i in indices:
        z = np.load(Path(out_dir) / f"{i:03d}.npz")
        out[int(i)] = {"means": z["means"], "vars": z["vars"], "n_samples": int(z["n_samples"])}
    return out
