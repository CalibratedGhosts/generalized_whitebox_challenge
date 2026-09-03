"""Network types, deterministic sampling, validity filtering and the manifest.

A network *type* is (width, depth, activation, strategy). The dataset is built
by repeatedly (1) drawing a type with each component uniform over its range,
(2) drawing a weight seed and building the weights, (3) keeping the network only
if a forward probe is well-conditioned (finite, final-layer RMS in [1e-2, 1e2]).
The whole procedure is a deterministic function of ``master_seed``.

Weights are NOT committed to the repository (they are ~800 MB of random
float32). Instead the manifest records each network's type, weight seed and a
SHA-256 of its weights; :func:`load_networks` regenerates the weights from the
seed and verifies the hash, so a mismatch (e.g. a different BLAS producing a
different QR) fails loudly instead of silently changing the task.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from gwc.activations import NAMES as ACTIVATIONS, apply_np, gain
from gwc.weights import STRATEGIES, sample_weight

WIDTHS = (16, 24, 32, 40, 48, 56, 64, 72, 84, 96, 128, 160, 192, 256, 288, 320, 352, 384)
DEPTHS = (4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24)

N_NETWORKS = 512
N_TRAIN = 256  # indices [0, 256) -> "train", [256, 512) -> "test"
MASTER_SEED = 20260903

PROBE_SAMPLES = 1024
RMS_MIN, RMS_MAX = 1e-2, 1e2


@dataclass(frozen=True)
class NetType:
    width: int
    depth: int
    activation: str
    strategy: str

    def key(self) -> str:
        return f"{self.activation}-{self.strategy}-{self.width}x{self.depth}"


@dataclass
class Network:
    """A concrete network: its type, its weights and provenance.

    ``weights[l]`` is the (width, width) float32 matrix of layer ``l`` (0-based);
    the forward pass is ``a = phi(a @ weights[l])`` starting from ``a = x``.
    """

    index: int
    split: str
    ntype: NetType
    weights: List[np.ndarray]
    seed: int
    name: str
    weight_sha256: str = ""

    # Convenience accessors used by estimators.
    @property
    def width(self) -> int:
        return self.ntype.width

    @property
    def depth(self) -> int:
        return self.ntype.depth

    @property
    def activation(self) -> str:
        return self.ntype.activation

    @property
    def strategy(self) -> str:
        return self.ntype.strategy


def split_of(index: int) -> str:
    return "train" if index < N_TRAIN else "test"


def sample_type(rng: np.random.Generator) -> NetType:
    """Each component uniform over its range, independently."""
    return NetType(
        width=int(WIDTHS[rng.integers(len(WIDTHS))]),
        depth=int(DEPTHS[rng.integers(len(DEPTHS))]),
        activation=str(ACTIVATIONS[rng.integers(len(ACTIVATIONS))]),
        strategy=str(STRATEGIES[rng.integers(len(STRATEGIES))]),
    )


def build_weights(ntype: NetType, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    g = gain(ntype.activation)
    return [sample_weight(ntype.strategy, ntype.width, g, rng) for _ in range(ntype.depth)]


def forward_np(ntype: NetType, weights: Sequence[np.ndarray], x: np.ndarray) -> Optional[np.ndarray]:
    """Unmetered forward pass; returns None if any layer is non-finite."""
    a = x
    for w in weights:
        a = apply_np(ntype.activation, a @ w).astype(np.float32, copy=False)
        if not np.all(np.isfinite(a)):
            return None
    return a


def probe_status(ntype: NetType, weights: Sequence[np.ndarray], n: int = PROBE_SAMPLES, seed: int = 0) -> str:
    x = np.random.default_rng(seed).standard_normal((n, ntype.width), dtype=np.float32)
    a = forward_np(ntype, weights, x)
    if a is None:
        return "nan"
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    if rms < RMS_MIN:
        return "vanish"
    if rms > RMS_MAX:
        return "huge"
    return "ok"


def weights_sha256(weights: Sequence[np.ndarray]) -> str:
    h = hashlib.sha256()
    for w in weights:
        h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()


def _name(index: int, ntype: NetType) -> str:
    return f"{index:03d}-{ntype.key()}"


def sample_networks(n: int = N_NETWORKS, master_seed: int = MASTER_SEED, verbose: bool = False,
                    keep_weights: bool = True) -> List[Network]:
    """Deterministically sample ``n`` valid networks (type -> weights -> probe).

    ``keep_weights=False`` drops each network's weights after hashing (the
    manifest only needs the hash), keeping the build's memory footprint small.
    """
    rng = np.random.default_rng(master_seed)
    out: List[Network] = []
    rejected: Dict[str, int] = {}
    while len(out) < n:
        ntype = sample_type(rng)
        seed = int(rng.integers(0, 2**62))
        weights = build_weights(ntype, seed)
        st = probe_status(ntype, weights)
        if st != "ok":
            rejected[f"{ntype.key()}:{st}"] = rejected.get(f"{ntype.key()}:{st}", 0) + 1
            continue
        idx = len(out)
        sha = weights_sha256(weights)
        out.append(Network(idx, split_of(idx), ntype, weights if keep_weights else [], seed, _name(idx, ntype), sha))
        if verbose and idx % 64 == 0:
            print(f"  sampled {idx + 1}/{n}", flush=True)
    if verbose and rejected:
        print(f"  rejected (resampled) degenerate draws: {rejected}")
    return out


# ---------------------------------------------------------------------------
# Manifest (committed) + regeneration/verification
# ---------------------------------------------------------------------------


def to_manifest(nets: Sequence[Network]) -> dict:
    return {
        "format": "gwc-manifest-v1",
        "master_seed": MASTER_SEED,
        "n_networks": len(nets),
        "n_train": N_TRAIN,
        "widths": list(WIDTHS),
        "depths": list(DEPTHS),
        "activations": list(ACTIVATIONS),
        "strategies": list(STRATEGIES),
        "networks": [
            {
                "index": n.index,
                "split": n.split,
                "name": n.name,
                "width": n.ntype.width,
                "depth": n.ntype.depth,
                "activation": n.ntype.activation,
                "strategy": n.ntype.strategy,
                "seed": n.seed,
                "weight_sha256": n.weight_sha256,
            }
            for n in nets
        ],
    }


def save_manifest(nets: Sequence[Network], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_manifest(nets), indent=1))


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_networks(manifest_path: Path, split: Optional[str] = None, indices: Optional[Sequence[int]] = None,
                  verify: bool = True) -> List[Network]:
    """Regenerate networks from the manifest (and verify their weight hashes)."""
    m = load_manifest(manifest_path)
    nets: List[Network] = []
    want = set(indices) if indices is not None else None
    for e in m["networks"]:
        if split is not None and e["split"] != split:
            continue
        if want is not None and e["index"] not in want:
            continue
        ntype = NetType(e["width"], e["depth"], e["activation"], e["strategy"])
        weights = build_weights(ntype, e["seed"])
        sha = weights_sha256(weights)
        if verify and sha != e["weight_sha256"]:
            raise RuntimeError(
                f"weight regeneration mismatch for network {e['index']} ({e['name']}): "
                f"got sha256 {sha[:12]}..., manifest {e['weight_sha256'][:12]}... "
                "This machine's NumPy/BLAS does not reproduce the dataset's weights; "
                "the ground truth would not match. Use the same numpy version/BLAS as the "
                "dataset build (numpy==2.2.6) or obtain the weight archive."
            )
        nets.append(Network(e["index"], e["split"], ntype, weights, e["seed"], e["name"], sha))
    return nets
