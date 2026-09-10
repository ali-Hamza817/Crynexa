#!/usr/bin/env python3
"""End-to-end demonstration runs for the frontend walkthrough.

Takes trained attackers, draws a SECRET KEY THEY HAVE NEVER SEEN, encrypts
held-out test images under it, and reconstructs. Emits per-image metrics and
the actual key values, so the walkthrough shows a real attack rather than a
retelling of one.

    python demo_run.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from crynexa.chaos.cipher import CipherConfig, build_cipher, sample_cipher_keys
from crynexa.data.pairs import to_tensor
from crynexa.data.sources import get_source
from crynexa.metrics.recon import psnr, ssim, _centred_unit, prior_floor
from crynexa.models.unet import build_model
from crynexa.train import RunConfig, infer

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
N_POOL = 1000          # candidate pool for the retrieval rank
N_SHOW = 8             # images rendered in the walkthrough

# (run dir, headline, what it demonstrates)
CASES = [
    ("E2_scope_rowcol_permonly__keyagnostic__resunet_cifar",
     "Row/column permutation — BROKEN under an unseen key",
     "The cipher permutes row order and column order independently, which preserves the "
     "multiset of whole rows and columns under every key. That structure is "
     "key-independent, so an attacker that never saw this key still recovers the image."),
    ("E2_posctrl_fixedperm_only__keyagnostic__linearmixer_cifar",
     "Key-independent permutation — positive control",
     "This cipher's transform is identical for every key, so there is nothing key-specific "
     "to learn. The attack should and does recover the plaintext exactly. It proves the "
     "instrument can detect a leak."),
    ("E2_rounds2__keyagnostic__resunet_cifar",
     "Keyed permutation + keyed diffusion — RESISTS",
     "The same attacker, same training budget, against the baseline cipher. Its same-key "
     "ceiling is 1.000, so it demonstrably CAN invert this cipher when the key is fixed — "
     "it simply learns nothing that transfers to a new key."),
    ("E2_negctrl_aes__keyagnostic__resunet_cifar",
     "AES-128-CTR — negative control",
     "A cipher with no known structural weakness. The attack must fail completely; if it "
     "did not, the pipeline would be leaking plaintext somewhere."),
]


def png(arr, scale=4):
    import base64, io
    from PIL import Image
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    im = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    b = io.BytesIO(); im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def run_case(run_name, device):
    d = RUNS / run_name
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    cfg = RunConfig(**ck["config"])
    ccfg = CipherConfig(**cfg.cipher)

    in_ch = ({"bits": 24, "both": 27, "pixel": 3}[cfg.input_repr]
             + (2 if cfg.coords else 0) + (3 if cfg.prev_ch else 0))
    mkw = dict(cfg.model_kw); mkw.setdefault("in_ch", in_ch)
    if cfg.out_mode == "categorical":
        mkw.setdefault("out_ch", 768); mkw.setdefault("logits", True)
    model = build_model(cfg.model, img_size=32, **mkw)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    train_imgs = get_source(cfg.dataset, "train")
    test_imgs = get_source(cfg.eval_dataset or cfg.dataset, "test")

    # A FRESH key, drawn now, from the full key space. With unlimited key
    # diversity during training the collision probability is ~0, so this key
    # was never seen.
    rng = np.random.default_rng(20260910)
    key = sample_cipher_keys(ccfg, 1, rng)[0]

    idx = rng.choice(len(test_imgs), size=N_POOL, replace=False)
    plain = test_imgs[idx]
    cipher = build_cipher(ccfg)
    ciph = cipher.encrypt(plain, np.repeat(key[None, :], N_POOL, axis=0))

    pt = to_tensor(plain)
    pred = infer(model, ciph, device, input_repr=cfg.input_repr,
                 coords=cfg.coords, prev_ch=cfg.prev_ch, out_mode=cfg.out_mode)

    ps = psnr(pred, pt).numpy()
    ss = ssim(pred, pt).numpy()
    sim = (_centred_unit(pred) @ _centred_unit(pt).t())
    rank = (sim > sim.diag().unsqueeze(1)).sum(1).numpy() + 1   # 1 = best
    floor = prior_floor(train_imgs, pt, "cpu")

    # show a representative spread, not a cherry-picked best
    order = np.argsort(-sim.diag().numpy())
    picks = [order[int(i)] for i in
             np.linspace(0, min(len(order), 40) - 1, N_SHOW).round()]

    return {
        "run": run_name,
        "cipher": ccfg.tag(),
        "cipherCfg": cfg.cipher,
        "attacker": cfg.model,
        "dataset": cfg.dataset,
        "trainedKeys": "unlimited (fresh key per training sample)"
                        if cfg.n_keys is None else str(cfg.n_keys),
        "key": {"x0": float(key[0]), "p1": float(key[1]),
                "p2": float(key[2]), "p3": float(key[3])},
        "poolSize": N_POOL,
        "top1": float((rank == 1).mean()),
        "top10": float((rank <= 10).mean()),
        "medianRank": float(np.median(rank)),
        "meanPsnr": float(ps.mean()), "meanSsim": float(ss.mean()),
        "floorPsnr": float(floor["psnr"]),
        "gainDb": float(ps.mean() - floor["psnr"]),
        "samples": [{
            "plain": png(plain[i]), "cipher": png(ciph[i]),
            "recon": png((pred[i].numpy() * 255).astype(np.uint8)),
            "psnr": float(ps[i]), "ssim": float(ss[i]), "rank": int(rank[i]),
        } for i in picks],
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = []
    for run_name, title, why in CASES:
        if not (RUNS / run_name / "model.pt").is_file():
            print(f"skip (no checkpoint): {run_name}")
            continue
        r = run_case(run_name, device)
        r["title"] = title
        r["why"] = why
        out.append(r)
        print(f"{title}\n    top-1 {r['top1']:.3f}  median rank {r['medianRank']:.0f}"
              f"/{N_POOL}  psnr {r['meanPsnr']:.2f} (floor {r['floorPsnr']:.2f})")
    (ROOT / "web" / "demo.json").write_text(json.dumps(out))
    print(f"\nwrote web/demo.json ({(ROOT/'web'/'demo.json').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
