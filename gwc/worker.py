"""Subprocess worker: runs a submission over a list of networks.

Invoked by :mod:`gwc.grader` as ``python -m gwc.worker ARGS.json``. It loads the
submission, calls ``setup`` once (off-budget, with a timeout), then for every
network runs ``predict`` inside a flopscope ``BudgetContext`` (FLOP budget +
wall-time cap) with a SIGALRM backstop, and records the prediction, FLOPs
used, wall time and any failure. The ground-truth targets are never given to
this process; the parent scores.
"""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import flopscope
from flopscope import BudgetExhaustedError, TimeExhaustedError

from gwc.activations import NAMES as ACTIVATIONS
from gwc.budget import flop_budget
from gwc.netspec import DEPTHS, WIDTHS, load_networks
from gwc.sdk import API_VERSION, BaseEstimator, SetupContext
from gwc.weights import STRATEGIES


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def load_estimator(path: str, class_name: Optional[str]):
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("gwc_submission", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import estimator from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gwc_submission"] = mod
    sys.path.insert(0, str(p.parent))
    spec.loader.exec_module(mod)
    if class_name:
        cls = getattr(mod, class_name)
    else:
        cls = getattr(mod, "Estimator", None)
        if cls is None:
            cands = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, BaseEstimator) and v is not BaseEstimator]
            if len(cands) != 1:
                raise ImportError("submission must define class Estimator(BaseEstimator) (or pass class_name)")
            cls = cands[0]
    return cls()


def main(args_path: str) -> int:
    a = json.loads(Path(args_path).read_text())
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    nets = load_networks(a["manifest"], indices=a["indices"])
    order = {i: k for k, i in enumerate(a["indices"])}
    nets.sort(key=lambda n: order[n.index])
    results = {}
    preds = {}

    signal.signal(signal.SIGALRM, _alarm)
    setup_err = None
    est = None
    try:
        signal.alarm(int(a["setup_timeout_s"]))
        est = load_estimator(a["estimator"], a.get("class_name"))
        ctx = SetupContext(API_VERSION, a["submission_dir"], int(a["seed"]), int(a["n_ref"]),
                           tuple(ACTIVATIONS), tuple(STRATEGIES), tuple(WIDTHS), tuple(DEPTHS))
        est.setup(ctx)
    except _Timeout:
        setup_err = f"setup timed out after {a['setup_timeout_s']}s"
    except BaseException as e:  # noqa: BLE001 - report everything
        setup_err = f"load/setup raised {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
    finally:
        signal.alarm(0)
    if setup_err:
        log(f"[worker] {setup_err}")

    t_all = time.time()
    for k, net in enumerate(nets):
        budget = flop_budget(net.ntype, int(a["n_ref"]))
        rec = {"index": net.index, "budget": int(budget), "flops_used": 0, "wall_s": 0.0,
               "failed": False, "failure": "", "error": ""}
        if setup_err:
            rec.update(failed=True, failure="error", error=setup_err)
            results[net.index] = rec
            continue
        pred = None
        bc = None
        t0 = time.time()
        try:
            signal.alarm(int(a["predict_timeout_s"]) + 5)
            with flopscope.BudgetContext(flop_budget=int(budget), wall_time_limit_s=float(a["predict_timeout_s"]),
                                         quiet=True) as bc:
                out = est.predict(net, int(budget))
                pred = np.array(np.asarray(out), dtype=np.float32, copy=True)
            rec["flops_used"] = int(bc.flops_used)
        except BudgetExhaustedError as e:
            rec.update(failed=True, failure="budget_exhausted", error=str(e)[:240])
            rec["flops_used"] = int(getattr(bc, "flops_used", budget) or budget)
        except (TimeExhaustedError, _Timeout) as e:
            rec.update(failed=True, failure="timeout", error=f"{type(e).__name__}: predict exceeded {a['predict_timeout_s']}s")
            rec["flops_used"] = int(getattr(bc, "flops_used", 0) or 0)
        except BaseException as e:  # noqa: BLE001
            rec.update(failed=True, failure="error",
                       error=f"{type(e).__name__}: {str(e)[:240]}\n{traceback.format_exc()[-1000:]}")
            rec["flops_used"] = int(getattr(bc, "flops_used", 0) or 0)
        finally:
            signal.alarm(0)
        rec["wall_s"] = time.time() - t0
        if not rec["failed"]:
            if pred is None or pred.shape != (net.depth, net.width) or not np.all(np.isfinite(pred)):
                rec.update(failed=True, failure="bad_output",
                           error=f"expected finite float array of shape {(net.depth, net.width)}, got {None if pred is None else pred.shape}")
            else:
                preds[net.index] = pred
        results[net.index] = rec
        if (k + 1) % 8 == 0 or k + 1 == len(nets):
            log(f"[worker] {k+1}/{len(nets)} networks done ({time.time()-t_all:.0f}s)")

    if est is not None:
        try:
            est.teardown()
        except BaseException:  # noqa: BLE001
            pass
    np.savez(a["out_npz"], **{f"p{i}": p for i, p in preds.items()})
    Path(a["out_json"]).write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
