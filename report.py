#!/usr/bin/env python3
"""Aggregate run results into tables and figures.

    python report.py --pattern 'E2_*' --out reports/E2
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def load(pattern):
    out = []
    for d in sorted(RUNS.glob(pattern)):
        f = d / "result.json"
        if f.is_file():
            try:
                out.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                pass
    return out


def row(r):
    s, u, fl, mm = r["seen"], r["unseen"], r["floor"], r["mismatch_control"]
    return {
        "run": r["config"]["name"],
        "cipher": r["cipher_tag"],
        "n_keys": r["config"]["n_keys"] or "inf",
        "model": r["config"]["model"],
        "dataset": r["config"]["dataset"],
        "seen_top1_100": s["top1_pool100"],
        "unseen_top1_100": u["top1_pool100"],
        "mismatch_top1_100": mm["top1_pool100"],
        "chance_100": s["chance_pool100"],
        "seen_psnr": s["psnr"],
        "unseen_psnr": u["psnr"],
        "floor_psnr": fl["psnr"],
        "unseen_psnr_gain": u["psnr"] - fl["psnr"],
        "seen_ssim": s["ssim"],
        "unseen_ssim": u["ssim"],
        "kgg_psnr": r["kgg_psnr"],
        "steps": r["config"]["steps"],
    }


def table(rows, cols=None, sort=None):
    cols = cols or list(rows[0].keys())
    if sort:
        rows = sorted(rows, key=lambda r: r.get(sort, 0), reverse=True)
    w = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(w[c]) for c in cols),
             "  ".join("-" * w[c] for c in cols)]
    for r in rows:
        lines.append("  ".join(fmt(r.get(c)).ljust(w[c]) for c in cols))
    return "\n".join(lines)


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def markdown(rows, cols):
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def plot_key_diversity(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def k(r):
        return 1e9 if r["n_keys"] == "inf" else int(r["n_keys"])
    rows = sorted(rows, key=k)
    xs = [k(r) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogx(xs, [r["seen_top1_100"] for r in rows], "o-", label="seen keys")
    ax[0].semilogx(xs, [r["unseen_top1_100"] for r in rows], "s-", label="unseen keys")
    ax[0].axhline(rows[0]["chance_100"], ls="--", c="k", lw=1, label="chance (1/100)")
    ax[0].set_xlabel("number of distinct training keys")
    ax[0].set_ylabel("top-1 retrieval @ pool 100")
    ax[0].set_title("Key diversity vs attack success")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].semilogx(xs, [r["seen_psnr"] for r in rows], "o-", label="seen keys")
    ax[1].semilogx(xs, [r["unseen_psnr"] for r in rows], "s-", label="unseen keys")
    ax[1].axhline(rows[0]["floor_psnr"], ls="--", c="k", lw=1, label="prior floor")
    ax[1].set_xlabel("number of distinct training keys")
    ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_title("Same metric in PSNR")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_cipher_strength(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["unseen_top1_100"], reverse=True)
    names = [r["run"].replace("E2_", "").replace("_noise", "").replace("_cifar", "")
             for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6))
    ax.barh(y, [r["unseen_top1_100"] for r in rows], color="#2b6cb0")
    ax.axvline(rows[0]["chance_100"], ls="--", c="r", lw=1.2, label="chance (1/100)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("unseen-key top-1 retrieval @ pool 100")
    ax.set_title("Key-agnostic attack success by cipher configuration")
    ax.legend(); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_samples(run_dir, path, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(run_dir / "samples.npz")
    fig, axes = plt.subplots(3, n, figsize=(1.35 * n, 4.4))
    for i in range(n):
        for j, (k, lab) in enumerate([("plain", "plaintext"), ("cipher", "ciphertext"),
                                      ("recon", "reconstruction")]):
            im = d[k][i]
            if im.shape[0] == 3:
                im = im.transpose(1, 2, 0)
            axes[j, i].imshow(im); axes[j, i].axis("off")
            if i == 0:
                axes[j, i].set_ylabel(lab)
                axes[j, i].text(-0.35, 0.5, lab, transform=axes[j, i].transAxes,
                                rotation=90, va="center", ha="center", fontsize=9)
    fig.suptitle(run_dir.name)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", default="reports/latest")
    ap.add_argument("--plot", choices=["keydiv", "strength", "none"], default="none")
    a = ap.parse_args()

    res = load(a.pattern)
    if not res:
        print(f"no results matching {a.pattern}")
        return
    rows = [row(r) for r in res]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    cols = ["run", "n_keys", "seen_top1_100", "unseen_top1_100",
            "mismatch_top1_100", "chance_100", "seen_psnr", "unseen_psnr",
            "floor_psnr", "kgg_psnr"]
    txt = table(rows, cols, sort="unseen_top1_100")
    print(txt)
    (out / "table.txt").write_text(txt)
    (out / "table.md").write_text(markdown(rows, cols))
    (out / "rows.json").write_text(json.dumps(rows, indent=2))

    if a.plot == "keydiv":
        plot_key_diversity(rows, out / "curve1_key_diversity.png")
    elif a.plot == "strength":
        plot_cipher_strength(rows, out / "curve3_cipher_strength.png")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
