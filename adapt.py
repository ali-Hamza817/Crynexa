#!/usr/bin/env python3
"""Few-shot key adaptation -- the well-posed attack setting (Curve 2).

Zero-shot key-agnostic reconstruction is information-theoretically bounded: with
the key marginalised out, a sound cipher leaves the posterior over plaintexts
equal to the prior. Supplying `n` known plaintext-ciphertext pairs under the
*target* key restores identifiability, which is also the classical
known-plaintext attack model.

The scientific payload is the comparison between:
  * a model pre-trained across a large key population, and
  * the same architecture trained from scratch,
both adapted to the same unseen key with the same `n`. A sample-efficiency
advantage for the pre-trained model is positive evidence that it learned
key-independent structure of the cipher family -- which the zero-shot setting
cannot show.

    python adapt.py --ckpt runs/E1_nkeys_inf/model.pt --device cuda:0
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from crynexa.chaos.cipher import CipherConfig, build_cipher, sample_cipher_keys
from crynexa.data.pairs import to_tensor
from crynexa.data.sources import get_source
from crynexa.metrics.recon import evaluate, prior_floor
from crynexa.models.unet import build_model
from crynexa.train import RunConfig, encode_input, infer, loss_fn

ROOT = Path(__file__).resolve().parent


def make_key_dataset(images, ccfg, key, idx):
    cipher = build_cipher(ccfg)
    keys = np.repeat(key[None, :], len(idx), axis=0)
    plain = images[idx]
    return plain, cipher.encrypt(plain, keys)


def finetune(model, sup_c, sup_p, cfg, device, steps, lr, bs=64):
    model = copy.deepcopy(model).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(sup_c)
    ct = to_tensor(sup_c)
    pt = to_tensor(sup_p)
    g = np.random.default_rng(0)
    for s in range(steps):
        i = g.integers(0, n, size=min(bs, n))
        xc = encode_input(ct[i].to(device), cfg.input_repr, cfg.coords)
        xp = pt[i].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
            pred = model(xc)
            loss = loss_fn(pred.float(), xp, cfg.loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model pre-trained over many keys")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-support", default="1,10,100,1000,10000")
    ap.add_argument("--n-keys-eval", type=int, default=3, help="distinct target keys")
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/E4_fewshot")
    a = ap.parse_args()

    device = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = RunConfig(**ck["config"])
    ccfg = CipherConfig(**cfg.cipher)

    train_imgs = get_source(cfg.dataset, "train")
    test_imgs = get_source(cfg.eval_dataset or cfg.dataset, "test")
    img_size = train_imgs.shape[1]
    in_ch = {"bits": 24, "both": 27, "pixel": 3}[cfg.input_repr] + (2 if cfg.coords else 0)

    def fresh_model():
        return build_model(cfg.model, img_size=img_size, in_ch=in_ch, **cfg.model_kw)

    pretrained = fresh_model()
    pretrained.load_state_dict(ck["model"])

    rng = np.random.default_rng(20240909)
    supports = [int(x) for x in a.n_support.split(",")]
    results = []

    for ki in range(a.n_keys_eval):
        # a target key never seen during pre-training (fresh uniform draw)
        key = sample_cipher_keys(ccfg, 1, rng)[0]
        q_idx = rng.choice(len(test_imgs), size=a.eval_n, replace=False)
        q_p, q_c = make_key_dataset(test_imgs, ccfg, key, q_idx)
        q_pt = to_tensor(q_p)
        floor = prior_floor(train_imgs, q_pt, "cpu")

        m0 = evaluate(infer(pretrained.to(device), q_c, device,
                            input_repr=cfg.input_repr, coords=cfg.coords), q_pt)
        results.append(dict(key_idx=ki, n_support=0, init="pretrained",
                            **{k: v for k, v in m0.items()}, floor_psnr=floor["psnr"]))
        print(f"[key{ki}] n=0     pretrained  top1@100 {m0['top1_pool100']:.4f} "
              f"psnr {m0['psnr']:.2f} (floor {floor['psnr']:.2f})", flush=True)

        for n in supports:
            s_idx = rng.choice(len(train_imgs), size=n, replace=False)
            s_p, s_c = make_key_dataset(train_imgs, ccfg, key, s_idx)
            for init in ["pretrained", "scratch"]:
                base = pretrained if init == "pretrained" else fresh_model()
                m = finetune(base, s_c, s_p, cfg, device, a.steps, a.lr)
                mm = evaluate(infer(m, q_c, device, input_repr=cfg.input_repr,
                                    coords=cfg.coords), q_pt)
                results.append(dict(key_idx=ki, n_support=n, init=init,
                                    **mm, floor_psnr=floor["psnr"]))
                print(f"[key{ki}] n={n:<6d} {init:11s} top1@100 "
                      f"{mm['top1_pool100']:.4f} psnr {mm['psnr']:.2f}", flush=True)
                del m
                torch.cuda.empty_cache()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
