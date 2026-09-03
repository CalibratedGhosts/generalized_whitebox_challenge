"""gwc command line.

  gwc smoke   --estimator PATH               fast check on 8 small train networks (unlimited)
  gwc submit  --estimator PATH [--split train|test] [--json] [--indices 3,7,9]
  gwc status                                 test-split cooldown, dataset summary
  gwc info    [--index N]                    dataset / type / budget information
  gwc build-targets                          (operator) encrypt data/cache/gt -> data/targets_*.enc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gwc import grader
from gwc.activations import NAMES as ACTIVATIONS, describe as describe_act
from gwc.budget import GROUND_TRUTH_SAMPLES, N_REF, flop_budget
from gwc.netspec import DEPTHS, WIDTHS, N_TRAIN
from gwc.weights import STRATEGIES, describe as describe_strategy


def _print_result(res, as_json: bool) -> None:
    if as_json:
        print(json.dumps(res, indent=1, default=float))
    else:
        print(grader.report(res))
        print(f"  full result (per-network rows): {res['run_dir']}/result.json")


def cmd_submit(a) -> int:
    try:
        res = grader.grade(a.estimator, a.split, indices=_parse_indices(a.indices), smoke=False,
                           predict_timeout_s=a.timeout, setup_timeout_s=a.setup_timeout, seed=a.seed,
                           force=a.force, class_name=a.class_name, quiet=a.json, tag=a.tag)
    except grader.CooldownError as e:
        print(f"[gwc] REFUSED: {e}", file=sys.stderr)
        return 2
    _print_result(res, a.json)
    if a.split == "test" and not a.json:
        print(f"  NOTE: test split graded. Next test grade allowed in {grader.TEST_COOLDOWN_S/3600:.0f} h.")
    return 0


def cmd_smoke(a) -> int:
    res = grader.grade(a.estimator, "train", smoke=True, predict_timeout_s=a.timeout,
                       setup_timeout_s=a.setup_timeout, seed=a.seed, class_name=a.class_name, quiet=a.json, tag=a.tag)
    _print_result(res, a.json)
    return 0


def cmd_status(a) -> int:
    rem = grader.cooldown_remaining_s()
    print("gwc status")
    print(f"  test-split cooldown : {'READY (you may grade on test now)' if rem <= 0 else f'{rem/60:.0f} min remaining'}")
    st = grader._read_state()
    if st:
        print(f"  test submissions so far: {len(st)}")
        for r in st[-5:]:
            print(f"    {r['utc'][:19]}  geo_adj={r['geo_adjusted_ratio']:.4g}  geo_ratio={r['geo_ratio']:.4g}  failed={r['n_failed']}  {Path(r['estimator']).name}")
    for split in ("train", "test"):
        print(f"  targets[{split}] present: {grader.TARGETS[split].exists()}")
    return 0


def cmd_info(a) -> int:
    nets = grader.load_meta()
    print("Generalized WhiteBox Challenge -- dataset")
    print(f"  networks: {len(nets)}  (train: {N_TRAIN}, test: {len(nets)-N_TRAIN})")
    print(f"  widths  : {list(WIDTHS)}")
    print(f"  depths  : {list(DEPTHS)}")
    print("  activations:")
    for n in ACTIVATIONS:
        print("    " + describe_act(n))
    print("  weight strategies:")
    for s in STRATEGIES:
        print("    " + describe_strategy(s))
    print(f"  budget per network: N_REF={N_REF:,} metered MC samples of that network; ground truth G={GROUND_TRUTH_SAMPLES:,} samples")
    if a.index is not None:
        n = next(x for x in nets if x.index == a.index)
        print(f"  network #{n.index}: {n.name}  split={n.split}  budget={flop_budget(n.ntype):,} FLOPs")
    else:
        smoke = grader.smoke_indices()
        print(f"  smoke set (train): {smoke}")
    return 0


def cmd_build_targets(a) -> int:
    """Operator only: bundle + encrypt the precomputed ground truth per split."""
    import numpy as np
    from gwc.crypto import encrypt_targets, load_or_create_key
    gt_dir = Path(a.gt_dir)
    key = load_or_create_key()
    nets = grader.load_meta()
    for split in ("train", "test"):
        idx = [n.index for n in nets if n.split == split]
        targets = {}
        for i in idx:
            z = np.load(gt_dir / f"{i:03d}.npz")
            targets[i] = {"means": z["means"], "vars": z["vars"], "n_samples": int(z["n_samples"])}
        blob = encrypt_targets(targets, key)
        grader.TARGETS[split].write_bytes(blob)
        print(f"  wrote {grader.TARGETS[split]} ({len(blob):,} bytes, {len(idx)} networks)")
    return 0


def _parse_indices(s):
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _add_common(p):
    p.add_argument("--estimator", required=True, help="estimator.py (or a folder containing estimator.py)")
    p.add_argument("--class-name", default=None)
    p.add_argument("--timeout", type=float, default=300.0, help="per-network predict wall-time cap (s)")
    p.add_argument("--setup-timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="label appended to the run directory name")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gwc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="grade an estimator on the train (unlimited) or test (once / 4h) split")
    _add_common(s)
    s.add_argument("--split", choices=["train", "test"], default="train")
    s.add_argument("--indices", default=None, help="comma-separated network indices (train split only, for debugging)")
    s.add_argument("--force", action="store_true", help="operator only: bypass the test cooldown")
    s.set_defaults(func=cmd_submit)
    s = sub.add_parser("smoke", help="fast sanity run on 8 small train networks")
    _add_common(s)
    s.set_defaults(func=cmd_smoke)
    s = sub.add_parser("status", help="cooldown + dataset status")
    s.set_defaults(func=cmd_status)
    s = sub.add_parser("info", help="dataset / activation / budget information")
    s.add_argument("--index", type=int, default=None)
    s.set_defaults(func=cmd_info)
    s = sub.add_parser("build-targets", help="(operator) encrypt precomputed ground truth")
    s.add_argument("--gt-dir", default=str(grader.ROOT / "data" / "cache" / "gt"))
    s.set_defaults(func=cmd_build_targets)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
