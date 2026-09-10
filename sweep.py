#!/usr/bin/env python3
"""Run an experiment matrix across available GPUs.

    python sweep.py --suite E1 --gpus 0,2
"""
import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

BASE = dict(dataset="cifar10", steps=6000, eval_every=1000, eval_n=2000,
            batch_size=256, workers=6, input_repr="both", coords=True, loss="l1")

# Attack success is a joint property of cipher and attacker inductive bias, so
# every cipher is attacked by both. Convolution cannot express a global pixel
# gather; a dense map can, but cannot exploit local structure as cheaply.
ATTACKERS = {
    "resunet": dict(model="resunet", input_repr="both", coords=True, model_kw={}),
    "linearmixer": dict(model="linearmixer", input_repr="pixel", coords=False,
                        model_kw={"pos_embed": 0}, lr=5e-4),
}

# Baseline uses feedback="none". CBC feedback pins the same-key ceiling at
# chance for every attacker tried (measured: XOR 1.000 vs CBC 0.011, all else
# identical), so using it as the baseline makes every OTHER axis -- rounds,
# permutation scope, precision -- unmeasurable. It is kept as an explicit
# variant so the effect is reported rather than silently poisoning the sweep.
DEFAULT_CIPHER = dict(name="chaos", map="logistic", rounds=2,
                      mode="permute_diffuse", perm_scope="full2d",
                      perm_keyed=True, feedback="none", precision="float64")


def cip(**kw):
    d = dict(DEFAULT_CIPHER)
    d.update(kw)
    return d


def suite_E1():
    """Curve 1 -- key diversity vs unseen-key attack success.

    n_keys=1 is the same-key baseline; n_keys=None is unlimited key diversity,
    i.e. the pure key-agnostic setting.
    """
    out = []
    for nk in [1, 4, 16, 64, 256, 1024, None]:
        for aname, akw in ATTACKERS.items():
            out.append(dict(BASE, **akw, n_keys=nk, cipher=cip(),
                            name=f"E1_nkeys_{nk if nk else 'inf'}__{aname}"))
    return out


def suite_E2():
    """Curve 3 -- which structural properties confer resistance.

    All runs use unlimited key diversity, so any success is genuinely
    key-agnostic. Includes a POSITIVE control (a fully key-independent cipher,
    which the attack MUST break) and the AES negative control (which it MUST
    NOT break). Between them they bracket the instrument's sensitivity.
    """
    variants = {
        # positive control: identical transform for every key
        "posctrl_fixedperm_only": cip(mode="permute_only", perm_keyed=False, rounds=1),
        # negative control
        "negctrl_aes": dict(name="aes"),
        # rounds
        "rounds1": cip(rounds=1),
        "rounds2": cip(rounds=2),
        "rounds3": cip(rounds=3),
        "rounds4": cip(rounds=4),
        # component isolation
        "permute_only_r1": cip(mode="permute_only", rounds=1),
        "permute_only_r2": cip(mode="permute_only", rounds=2),
        "diffuse_only_r1": cip(mode="diffuse_only", rounds=1),
        "diffuse_only_r1_cbc": cip(mode="diffuse_only", rounds=1, feedback="cbc"),
        # permutation scope
        "scope_rowcol": cip(perm_scope="rowcol"),
        "scope_blockwise64": cip(perm_scope="blockwise", block_size=64),
        "scope_rowcol_permonly": cip(perm_scope="rowcol", mode="permute_only", rounds=1),
        # key-independent permutation, keyed diffusion
        "fixedperm_keyed_diff": cip(perm_keyed=False),
        # diffusion structure
        "feedback_cbc": cip(feedback="cbc"),
        # keystream precision degradation
        "float32": cip(precision="float32"),
    }
    # Every variant is run at BOTH key regimes. n_keys=1 gives the same-key
    # ceiling for that cipher: the best this attacker can do when the key is
    # fixed. Without it, "attack fails on unseen keys" is uninterpretable --
    # it cannot be told apart from "this attacker cannot invert this cipher at
    # all". The key-agnostic number is only meaningful relative to its ceiling.
    out = []
    for k, v in variants.items():
        for regime, nk in [("samekey", 1), ("keyagnostic", None)]:
            for aname, akw in ATTACKERS.items():
                out.append(dict(BASE, **akw, cipher=v, n_keys=nk,
                                name=f"E2_{k}__{regime}__{aname}"))
    return out


def suite_E3():
    """Curve 4 -- transfer across chaos map and image distribution."""
    out = []
    for m in ["logistic", "tent", "henon", "pwlcm"]:
        out.append(dict(BASE, name=f"E3_map_{m}", n_keys=None, cipher=cip(map=m)))
    for ds in ["noise"]:
        out.append(dict(BASE, name=f"E3_data_{ds}", n_keys=None, dataset=ds,
                        cipher=cip()))
    # architecture comparison on the permutation-dominant case
    for mdl in ["resunet", "cnn", "pixelmixer"]:
        out.append(dict(BASE, name=f"E3_arch_{mdl}_permonly", n_keys=None,
                        model=mdl, batch_size=128 if mdl == "pixelmixer" else 256,
                        cipher=cip(mode="permute_only", perm_keyed=False, rounds=1)))
    return out


SUITES = {"E1": suite_E1, "E2": suite_E2, "E3": suite_E3}


def worker(gpu, q, results, lock):
    while True:
        try:
            cfg = q.get_nowait()
        except queue.Empty:
            return
        name = cfg["name"]
        log = RUNS / f"{name}.log"
        RUNS.mkdir(exist_ok=True)
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.call(
                ["python3", str(ROOT / "run_experiment.py"),
                 "--name", name, "--device", f"cuda:{gpu}",
                 "--json", json.dumps(cfg)],
                stdout=fh, stderr=subprocess.STDOUT)
        with lock:
            results.append((name, rc, time.time() - t0))
            print(f"[gpu{gpu}] {name} rc={rc} {time.time()-t0:.0f}s", flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--gpus", default="0,2")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--only", help="comma-separated substring filter on run names")
    ap.add_argument("--dataset", help="override plaintext source for the whole suite")
    ap.add_argument("--tag", default="", help="suffix appended to every run name")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip runs that already have a result.json (resume)")
    a = ap.parse_args()

    cfgs = SUITES[a.suite]()
    if a.steps:
        for c in cfgs:
            c["steps"] = a.steps
    if a.dataset:
        for c in cfgs:
            c["dataset"] = a.dataset
    if a.tag:
        for c in cfgs:
            c["name"] = c["name"] + a.tag
    if a.only:
        pats = a.only.split(",")
        cfgs = [c for c in cfgs if any(p in c["name"] for p in pats)]

    if a.skip_done:
        before = len(cfgs)
        cfgs = [c for c in cfgs if not (RUNS / c["name"] / "result.json").is_file()]
        print(f"skip-done: {before - len(cfgs)} already complete, {len(cfgs)} to run",
              flush=True)

    q = queue.Queue()
    for c in cfgs:
        q.put(c)
    print(f"suite {a.suite}: {len(cfgs)} runs on GPUs {a.gpus}", flush=True)

    results, lock = [], threading.Lock()
    ts = [threading.Thread(target=worker, args=(g, q, results, lock))
          for g in a.gpus.split(",")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"DONE {a.suite}: {sum(1 for _, rc, _ in results if rc == 0)}/{len(results)} ok",
          flush=True)


if __name__ == "__main__":
    main()
