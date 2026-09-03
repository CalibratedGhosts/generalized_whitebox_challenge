"""Per-network scores, aggregation across heterogeneous network types, and reports.

Why not raw MSE?  The 8 activations have wildly different output scales
(deep ``cos`` activations have variance ~2e-4, ``tanh_rmsnorm`` ~0.4; odd
activations have ~zero means). Averaging raw MSEs would let one activation
dominate. Instead every network is scored **relative to budget-matched
Monte-Carlo sampling**:

    ratio = mse_final / (sigma^2 / N_REF)

where ``sigma^2`` is the mean final-layer activation variance of that network
(so ``sigma^2 / N_REF`` is exactly the MSE a full-budget MC estimator gets).
``ratio < 1`` means "better than sampling", the same bar for every type.
The challenge's compute discount is then applied:

    adjusted_ratio = ratio * max(0.1, flops_used / budget)     (x1.0 on failure)

and the headline number is the **geometric mean** of ``adjusted_ratio`` over
networks (scale-free; no single network can dominate). Lower is better.

Also reported: the bias-corrected ratio ``(mse - sigma^2/G) / (sigma^2/N_REF)``
(the stored ground truth carries MC noise ``sigma^2/G`` which inflates every
measured MSE by that amount), the worst-decile geometric mean (generalisation
tail), the fraction of networks beating sampling, and per-activation /
per-strategy / per-size breakdowns.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from gwc.budget import N_REF, score_multiplier
from gwc.netspec import Network

EPS = 1e-30


@dataclass
class NetScore:
    index: int
    name: str
    split: str
    activation: str
    strategy: str
    width: int
    depth: int
    budget: int
    flops_used: int
    utilization: float
    wall_s: float
    failed: bool
    failure: str          # "", "error", "timeout", "budget_exhausted", "bad_output"
    error: str
    mse_final: float
    mse_all: float
    mc_ref_final: float   # sigma^2 / N_REF   (full-budget sampling MSE, final layer)
    mc_ref_all: float
    noise_floor_final: float  # sigma^2 / G  (ground-truth MC noise)
    ratio_final: float
    ratio_all: float
    ratio_final_corrected: float
    at_noise_floor: bool
    multiplier: float
    adjusted_ratio: float
    beats_sampling: bool


def score_network(net: Network, target: Dict[str, np.ndarray], pred: Optional[np.ndarray],
                  stats: Dict, n_ref: int = N_REF) -> NetScore:
    means = np.asarray(target["means"], dtype=np.float64)
    variances = np.asarray(target["vars"], dtype=np.float64)
    n_gt = int(target["n_samples"])
    d, w = means.shape
    failed = bool(stats.get("failed", False))
    failure = str(stats.get("failure", ""))
    error = str(stats.get("error", ""))
    if not failed:
        if pred is None:
            failed, failure, error = True, "bad_output", "no prediction returned"
        else:
            pred = np.asarray(pred, dtype=np.float64)
            if pred.shape != (d, w) or not np.all(np.isfinite(pred)):
                failed, failure = True, "bad_output"
                error = f"expected finite array of shape {(d, w)}, got shape {getattr(pred, 'shape', None)}"
    if failed:
        pred = np.zeros((d, w), dtype=np.float64)  # challenge semantics: output zeroed, no discount

    mse_final = float(np.mean((pred[-1] - means[-1]) ** 2))
    mse_all = float(np.mean((pred - means) ** 2))
    var_final = float(np.mean(variances[-1]))
    var_all = float(np.mean(variances))
    mc_ref_final = var_final / n_ref
    mc_ref_all = var_all / n_ref
    nf = var_final / n_gt
    ratio_final = mse_final / max(mc_ref_final, EPS)
    ratio_all = mse_all / max(mc_ref_all, EPS)
    ratio_corr = max(mse_final - nf, 0.0) / max(mc_ref_final, EPS)
    budget = int(stats.get("budget", 0))
    flops = int(stats.get("flops_used", 0))
    mult = float(score_multiplier(flops, budget, failed=failed))
    return NetScore(
        index=net.index, name=net.name, split=net.split, activation=net.activation,
        strategy=net.strategy, width=net.width, depth=net.depth, budget=budget,
        flops_used=flops, utilization=(flops / budget if budget else 0.0),
        wall_s=float(stats.get("wall_s", 0.0)), failed=failed, failure=failure, error=error,
        mse_final=mse_final, mse_all=mse_all, mc_ref_final=mc_ref_final, mc_ref_all=mc_ref_all,
        noise_floor_final=nf, ratio_final=ratio_final, ratio_all=ratio_all,
        ratio_final_corrected=ratio_corr, at_noise_floor=bool(mse_final < 2.0 * nf),
        multiplier=mult, adjusted_ratio=ratio_final * mult,
        beats_sampling=bool((not failed) and ratio_final < 1.0),
    )


def geo_mean(xs: Sequence[float]) -> float:
    xs = [max(float(x), EPS) for x in xs]
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs))) if xs else float("nan")


def _width_bucket(w: int) -> str:
    return "w<=64" if w <= 64 else ("w72-192" if w <= 192 else "w>=256")


def _depth_bucket(d: int) -> str:
    return "d<=8" if d <= 8 else ("d10-16" if d <= 16 else "d>=20")


def _group(rows: List[NetScore], keyf) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    groups: Dict[str, List[NetScore]] = {}
    for r in rows:
        groups.setdefault(keyf(r), []).append(r)
    for k in sorted(groups):
        g = groups[k]
        out[k] = {
            "n": len(g),
            "geo_adjusted_ratio": geo_mean([r.adjusted_ratio for r in g]),
            "geo_ratio": geo_mean([r.ratio_final for r in g]),
            "frac_beats_sampling": sum(r.beats_sampling for r in g) / len(g),
            "n_failed": sum(r.failed for r in g),
            "mean_utilization": float(np.mean([r.utilization for r in g])),
        }
    return out


def aggregate(rows: List[NetScore]) -> Dict:
    if not rows:
        return {"n": 0}
    adj = [r.adjusted_ratio for r in rows]
    srt = sorted(adj, reverse=True)
    k = max(1, len(srt) // 10)
    return {
        "n": len(rows),
        "n_failed": sum(r.failed for r in rows),
        "failures": {f: sum(1 for r in rows if r.failure == f) for f in ("error", "timeout", "budget_exhausted", "bad_output")},
        # ---- headline ----
        "geo_adjusted_ratio": geo_mean(adj),
        # ---- diagnostics ----
        "geo_ratio": geo_mean([r.ratio_final for r in rows]),
        "geo_ratio_corrected": geo_mean([r.ratio_final_corrected for r in rows]),
        "geo_ratio_all_layers": geo_mean([r.ratio_all for r in rows]),
        "median_adjusted_ratio": float(np.median(adj)),
        "mean_adjusted_ratio": float(np.mean(adj)),
        "worst_decile_geo_adjusted_ratio": geo_mean(srt[:k]),
        "frac_beats_sampling": sum(r.beats_sampling for r in rows) / len(rows),
        "frac_at_noise_floor": sum(r.at_noise_floor for r in rows) / len(rows),
        "mean_utilization": float(np.mean([r.utilization for r in rows])),
        "mean_multiplier": float(np.mean([r.multiplier for r in rows])),
        "total_wall_s": float(sum(r.wall_s for r in rows)),
        "by_activation": _group(rows, lambda r: r.activation),
        "by_strategy": _group(rows, lambda r: r.strategy),
        "by_width": _group(rows, lambda r: _width_bucket(r.width)),
        "by_depth": _group(rows, lambda r: _depth_bucket(r.depth)),
        "worst": [
            {"index": r.index, "name": r.name, "adjusted_ratio": r.adjusted_ratio, "ratio": r.ratio_final,
             "failure": r.failure, "error": r.error[:160]}
            for r in sorted(rows, key=lambda r: -r.adjusted_ratio)[:10]
        ],
        "best": [
            {"index": r.index, "name": r.name, "adjusted_ratio": r.adjusted_ratio, "ratio": r.ratio_final}
            for r in sorted(rows, key=lambda r: r.adjusted_ratio)[:5]
        ],
    }


def _f(x: float) -> str:
    return f"{x:9.3e}" if (x < 1e-2 or x >= 1e3) else f"{x:9.3f}"


def format_report(agg: Dict, title: str) -> str:
    L = []
    L.append("=" * 78)
    L.append(title)
    L.append("=" * 78)
    L.append(f"  networks: {agg['n']}   failed: {agg['n_failed']}  {agg['failures']}")
    L.append(f"  HEADLINE  geo-mean adjusted ratio : {_f(agg['geo_adjusted_ratio'])}   (lower is better; 1.0 = full-budget sampling)")
    L.append(f"            geo-mean raw ratio      : {_f(agg['geo_ratio'])}   (MSE / sampling MSE, before compute discount)")
    L.append(f"            geo-mean bias-corrected : {_f(agg['geo_ratio_corrected'])}   (ground-truth noise removed)")
    L.append(f"            all-layers ratio        : {_f(agg['geo_ratio_all_layers'])}")
    L.append(f"            worst-decile geo-mean   : {_f(agg['worst_decile_geo_adjusted_ratio'])}   (generalisation tail)")
    L.append(f"            median / mean adjusted  : {_f(agg['median_adjusted_ratio'])} / {_f(agg['mean_adjusted_ratio'])}")
    L.append(f"  beats sampling: {100*agg['frac_beats_sampling']:.1f}%   at noise floor: {100*agg['frac_at_noise_floor']:.1f}%   "
             f"mean budget use: {100*agg['mean_utilization']:.2f}%   mean multiplier: {agg['mean_multiplier']:.3f}   wall: {agg['total_wall_s']:.0f}s")
    for name, key in (("activation", "by_activation"), ("weight strategy", "by_strategy"), ("width", "by_width"), ("depth", "by_depth")):
        L.append("-" * 78)
        L.append(f"  by {name}:{'':<15} {'n':>4} {'geo adj':>10} {'geo ratio':>10} {'beats%':>7} {'fail':>5} {'use%':>7}")
        for k, g in agg[key].items():
            L.append(f"    {k:<26} {g['n']:>4} {_f(g['geo_adjusted_ratio']):>10} {_f(g['geo_ratio']):>10} "
                     f"{100*g['frac_beats_sampling']:>6.0f}% {g['n_failed']:>5} {100*g['mean_utilization']:>6.2f}%")
    L.append("-" * 78)
    L.append("  worst networks:")
    for r in agg["worst"][:6]:
        extra = f"  [{r['failure']}] {r['error']}" if r["failure"] else ""
        L.append(f"    #{r['index']:03d} {r['name']:<36} adj {_f(r['adjusted_ratio'])}  ratio {_f(r['ratio'])}{extra}")
    L.append("  best networks:")
    for r in agg["best"][:3]:
        L.append(f"    #{r['index']:03d} {r['name']:<36} adj {_f(r['adjusted_ratio'])}  ratio {_f(r['ratio'])}")
    L.append("=" * 78)
    return "\n".join(L)


def rows_to_dicts(rows: List[NetScore]) -> List[Dict]:
    return [asdict(r) for r in rows]
