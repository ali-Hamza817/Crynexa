#!/usr/bin/env python3
"""Generate web/data.json for the React results frontend.

Bundles measured run results, dataset provenance, cipher-validation statistics
and sample reconstructions into a single JSON payload.

    python build_web.py
"""
import base64
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
WEB = ROOT / "web"


# ------------------------------------------------------------------ util --
def png_b64(arr, scale=3):
    from PIL import Image
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    im = Image.fromarray(arr.astype(np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def parse_name(name):
    suite = name.split("_")[0]
    regime = ("same-key" if "__samekey" in name else
              "key-agnostic" if "__keyagnostic" in name else "-")
    attacker = next((a for a in ("resunet", "linearmixer", "pixelmixer", "cnn")
                     if a in name), "-")
    dataset = next((d for d in ("shapes", "cifar10", "noise", "stl10")
                    if name.endswith("_" + d) or f"_{d}_" in name), "-")
    variant = name
    for cut in ("__samekey", "__keyagnostic"):
        variant = variant.split(cut)[0]
    if variant.startswith(suite + "_"):
        variant = variant[len(suite) + 1:]
    return suite, variant, regime, attacker, dataset


SAMPLE_CACHE = ROOT / "reports" / "sample_cache"


def cached_samples(run_name, npz_path):
    """Encoded PNGs never change once a run has finished, so cache them.
    Without this every refresh re-encodes every run and the cost grows with
    the number of completed runs."""
    SAMPLE_CACHE.mkdir(parents=True, exist_ok=True)
    c = SAMPLE_CACHE / f"{run_name}.json"
    if c.is_file():
        try:
            return json.loads(c.read_text())
        except json.JSONDecodeError:
            pass
    z = np.load(npz_path)
    out = [{"plain": png_b64(z["plain"][i]), "cipher": png_b64(z["cipher"][i]),
            "recon": png_b64(z["recon"][i])} for i in range(min(6, len(z["plain"])))]
    c.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------ runs --
def collect_runs():
    rows, samples = [], {}
    for d in sorted(RUNS.iterdir()):
        f = d / "result.json"
        if not f.is_file():
            continue
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        c = r["config"]
        s, u, fl, mm = r["seen"], r["unseen"], r["floor"], r["mismatch_control"]
        suite, variant, regime, attacker, dataset = parse_name(c["name"])
        ratio = u["top1_pool100"] / max(s["chance_pool100"], 1e-9)
        rows.append({
            "run": c["name"], "suite": suite, "variant": variant, "regime": regime,
            "attacker": attacker if attacker != "-" else c["model"],
            "dataset": dataset if dataset != "-" else c["dataset"],
            "cipher": r["cipher_tag"], "cipherCfg": c["cipher"],
            "nKeys": c["n_keys"] if c["n_keys"] else "unlimited",
            "steps": c["steps"], "paramsM": round(r["n_params"] / 1e6, 1),
            "seenTop1": s["top1_pool100"], "unseenTop1": u["top1_pool100"],
            "mismatchTop1": mm["top1_pool100"], "chance": s["chance_pool100"],
            "seenTop1k": s.get("top1_pool1000"), "unseenTop1k": u.get("top1_pool1000"),
            "seenPsnr": s["psnr"], "unseenPsnr": u["psnr"], "floorPsnr": fl["psnr"],
            "gainDb": u["psnr"] - fl["psnr"],
            "seenSsim": s["ssim"], "unseenSsim": u["ssim"], "floorSsim": fl["ssim"],
            "kgg": None if r["kgg_psnr"] != r["kgg_psnr"] else r["kgg_psnr"],
            "chanceRatio": ratio,
            "verdict": "broken" if ratio >= 5 else "partial" if ratio >= 2 else "resisted",
            "secs": round(r["train_seconds"]),
            "history": [{"step": h["step"],
                         "seen": h.get("seen", {}).get("top1_pool100"),
                         "unseen": h["unseen"].get("top1_pool100"),
                         "unseenPsnr": h["unseen"]["psnr"]} for h in r["history"]],
        })
        npz = d / "samples.npz"
        if npz.is_file():
            try:
                samples[c["name"]] = cached_samples(c["name"], npz)
            except Exception:
                pass
    return rows, samples


# -------------------------------------------------------------- datasets --
DATASET_DOCS = {
    "cifar10": {
        "name": "CIFAR-10",
        "role": "Primary — natural images",
        "shape": "50,000 train / 10,000 test · 32×32×3",
        "source": "uoft-cs/cifar10 (HuggingFace mirror of Krizhevsky, 2009)",
        "why": "The resolution keeps the full experiment matrix affordable while the "
               "images carry real natural-image redundancy, which is the resource a "
               "neural attack actually consumes. It is also the benchmark the prior "
               "neural-cryptanalysis literature uses, so results are comparable.",
        "color": "indigo",
    },
    "shapes": {
        "name": "Shapes (synthetic)",
        "role": "Control — tunable redundancy",
        "shape": "50,000 train / 10,000 test · 32×32×3",
        "source": "Generated in-repo (coloured discs on coloured ground)",
        "why": "Natural-image-like spatial redundancy with none of the semantic content "
               "a model could memorise, and no download required. Sits between CIFAR-10 "
               "and noise on the redundancy axis, isolating structure from semantics.",
        "color": "cyan",
    },
    "noise": {
        "name": "Uniform noise",
        "role": "Control — zero redundancy",
        "shape": "50,000 train / 10,000 test · 32×32×3",
        "source": "Generated in-repo (uniform random bytes)",
        "why": "There is no image prior to exploit, so exact inversion is the only route "
               "to a correct answer. Any above-chance retrieval here is unambiguously "
               "cryptanalytic rather than learned image statistics — the sharpest "
               "available answer to RQ1.",
        "color": "amber",
    },
}

DATASET_STATS = [
    {"dataset": "uniform noise", "absDiff": 85.30, "corr": 0.0006},
    {"dataset": "shapes", "absDiff": 5.97, "corr": 0.9162},
]

ENTROPY_CEILING = [
    {"size": "32×32×3 (CIFAR-10)", "samples": 3072, "ceiling": 7.9407},
    {"size": "96×96×3 (STL-10)", "samples": 27648, "ceiling": 7.9932},
    {"size": "asymptotic", "samples": None, "ceiling": 8.0000},
]

METRIC_CHECK = [
    {"predictor": "perfect (pred = target)", "p10": 1.0000, "p100": 1.0000, "p1000": 1.0000},
    {"predictor": "constant (prior-only)", "p10": 0.1018, "p100": 0.0109, "p1000": 0.0018},
    {"predictor": "random", "p10": 0.1014, "p100": 0.0088, "p1000": 0.0001},
    {"predictor": "chance", "p10": 0.1, "p100": 0.01, "p1000": 0.001},
]

ATTACKERS = {
    "resunet": {"label": "ResUNet", "desc":
                "Residual U-Net with skip connections, coordinate channels and a "
                "learnable positional embedding. The standard image-to-image attacker.",
                "input": "pixel + bit-planes + coords (29ch)"},
    "linearmixer": {"label": "LinearMixer", "desc":
                    "Dense N×N map over the flattened image. Represents an arbitrary "
                    "fixed permutation exactly, with no locality assumption.",
                    "input": "pixel (3ch)"},
    "pixelmixer": {"label": "PixelMixer", "desc":
                   "Transformer over patch tokens — global mixing, permutation tolerant.",
                   "input": "pixel + bit-planes + coords"},
    "cnn": {"label": "PlainCNN", "desc":
            "Encoder-decoder with no skip connections. Ablation baseline.",
            "input": "pixel + bit-planes + coords"},
}


def cipher_stats():
    f = ROOT / "reports" / "cipher_stats.json"
    if not f.is_file():
        return []
    d = json.loads(f.read_text())
    out = []
    for k, v in d.items():
        out.append({"config": k, "entropy": v["entropy_cipher"],
                    "corrH": v["corr_h_cipher"], "corrV": v["corr_v_cipher"],
                    "npcr": v["npcr"], "uaci": v["uaci"],
                    "keySensNpcr": v["key_sens_npcr"]})
    return out


def pipeline_status():
    log = RUNS / "pipeline.log"
    lines = log.read_text().strip().splitlines() if log.is_file() else []
    suites = []
    for tag, f in [("E2 · shapes", "sweep_E2_shapes.log"),
                   ("E1 · CIFAR-10", "sweep_E1_cifar.log"),
                   ("E2 · CIFAR-10", "sweep_E2_cifar.log"),
                   ("E1 · shapes", "sweep_E1_shapes.log")]:
        p = RUNS / f
        if not p.is_file():
            suites.append({"suite": tag, "done": 0, "total": None, "state": "queued"})
            continue
        t = p.read_text()
        done = t.count("rc=")
        total = None
        for ln in t.splitlines():
            if ln.startswith("suite ") and " runs on" in ln:
                total = int(ln.split(":")[1].split("runs")[0].strip())
        state = "complete" if "DONE " in t else "running"
        suites.append({"suite": tag, "done": done, "total": total, "state": state})
    return {"log": lines[-12:], "suites": suites}


def tests_status(max_age=1800):
    """Cached: the suite takes ~30s, far longer than the refresh interval."""
    cache = ROOT / "reports" / "tests_cache.json"
    if cache.is_file():
        try:
            c = json.loads(cache.read_text())
            if (datetime.now().timestamp() - c.get("ts", 0)) < max_age:
                return {k: c[k] for k in ("passed", "output")}
        except Exception:
            pass
    try:
        r = subprocess.run(["python3", str(ROOT / "tests" / "test_cipher.py")],
                           capture_output=True, text=True, timeout=600)
        out = {"passed": r.returncode == 0,
               "output": [l for l in r.stdout.splitlines() if l.strip()]}
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps({**out, "ts": datetime.now().timestamp()}))
        return out
    except Exception as e:
        return {"passed": False, "output": [f"could not run: {e}"]}


def main():
    WEB.mkdir(exist_ok=True)
    rows, samples = collect_runs()
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "runs": rows,
        "samples": samples,
        "datasets": DATASET_DOCS,
        "datasetStats": DATASET_STATS,
        "entropyCeiling": ENTROPY_CEILING,
        "metricCheck": METRIC_CHECK,
        "attackers": ATTACKERS,
        "cipherStats": cipher_stats(),
        "pipeline": pipeline_status(),
        "tests": tests_status(),
    }
    p = WEB / "data.json"
    p.write_text(json.dumps(payload))
    print(f"{len(rows)} runs, {len(samples)} sample sets -> {p} "
          f"({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
