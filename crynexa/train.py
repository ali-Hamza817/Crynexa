"""Train one neural-cryptanalysis attacker and evaluate it with full controls."""
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .chaos.cipher import CipherConfig, build_cipher
from .data.pairs import (PairStream, make_fixed_eval, to_tensor,
                         to_bitplanes, coord_channels, prev_element)
from .data.sources import get_source
from .metrics.recon import evaluate, prior_floor, gain_over_floor, kgg, ssim
from .models.unet import build_model

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# offset that guarantees the evaluation key pool is disjoint from training
UNSEEN_KEY_SEED_OFFSET = 999_983


@dataclass
class RunConfig:
    name: str = "run"
    cipher: dict = field(default_factory=lambda: CipherConfig().to_dict())
    dataset: str = "cifar10"
    eval_dataset: str = None          # None -> same as `dataset`
    model: str = "resunet"
    model_kw: dict = field(default_factory=dict)
    n_keys: int = None                # None -> fresh key per sample (unlimited)
    batch_size: int = 256
    steps: int = 8000
    lr: float = 2e-4
    warmup_steps: int = 200
    eval_n: int = 2000
    eval_every: int = 1000
    seed: int = 0
    device: str = "cuda:0"
    workers: int = 4
    input_repr: str = "both"   # both | bits | pixel
    coords: bool = True        # positional channels (see coord_channels)
    prev_ch: bool = True       # flat-raster predecessor (see prev_element)
    out_mode: str = "regress"  # regress | categorical (256-way per byte)
    train_subset: int = None   # cap distinct plaintexts (memorisation axis)
    loss: str = "l1"
    amp: bool = True


def decode_logits(x, out_mode):
    """Categorical head -> image. Uses the posterior mean over byte values,
    which is the Bayes estimator under squared error and degrades gracefully
    when the model is uncertain (argmax would report hard errors instead)."""
    if out_mode != "categorical":
        return x
    B, _, H, W = x.shape
    p = x.reshape(B, 3, 256, H, W).float().softmax(2)
    vals = torch.arange(256, device=x.device, dtype=p.dtype).view(1, 1, 256, 1, 1) / 255.0
    return (p * vals).sum(2)


def categorical_loss(logits, target01):
    """Cross-entropy over 256 byte values per channel per pixel.

    The inverse of a CBC-style diffusion, p_i = ((c_i XOR k_i) - c_{i-1}) mod 256,
    contains a jump discontinuity at every pixel. A continuous regression head
    cannot represent that: a misplaced threshold costs a maximal +/-255 error.
    A categorical head represents it natively.
    """
    B, _, H, W = logits.shape
    l = logits.reshape(B, 3, 256, H, W).permute(0, 2, 1, 3, 4)
    tgt = (target01 * 255.0).round().clamp(0, 255).long()
    return F.cross_entropy(l.reshape(B, 256, -1).float(), tgt.reshape(B, -1))


def loss_fn(pred, target, kind):
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "mse":
        return F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    s = 1.0 - ssim(pred.float(), target.float()).mean()
    return l1 + 0.15 * s


def encode_input(x, repr_, coords=True, prev_ch=False):
    """Give the attacker an encoding adequate to the cipher's algebra.

    XOR is linear only in the bit-plane view; mod-256 addition/subtraction (the
    CBC feedback term) is near-linear only in the pixel-value view. An attacker
    given only one of them cannot express the other, and then fails for reasons
    of input encoding rather than cipher strength -- which would masquerade as
    a security result. "both" removes that confound.
    """
    if repr_ == "bits":
        h = to_bitplanes(x)
    elif repr_ == "both":
        h = torch.cat([x, to_bitplanes(x)], 1)
    else:
        h = x
    if prev_ch:
        h = torch.cat([h, prev_element(x)], 1)
    if coords:
        b, _, hh, ww = h.shape
        h = torch.cat([h, coord_channels(b, hh, ww, h.device, h.dtype)], 1)
    return h


@torch.no_grad()
def infer(model, ciph_u8, device, batch=512, input_repr="both", coords=True,
          prev_ch=False, out_mode="regress"):
    model.eval()
    outs = []
    for s in range(0, len(ciph_u8), batch):
        x = encode_input(to_tensor(ciph_u8[s:s + batch]).to(device, non_blocking=True),
                         input_repr, coords, prev_ch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            outs.append(decode_logits(model(x), out_mode).float().cpu())
    model.train()
    return torch.cat(outs)


def run(cfg: RunConfig, out_dir=None, verbose=True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    ccfg = CipherConfig(**cfg.cipher)

    train_imgs = get_source(cfg.dataset, "train")
    if cfg.train_subset:
        train_imgs = train_imgs[:cfg.train_subset]
    eval_src = cfg.eval_dataset or cfg.dataset
    test_imgs = get_source(eval_src, "test")
    img_size = train_imgs.shape[1]

    # ---- fixed evaluation sets ------------------------------------------
    # seen  : keys drawn from the SAME pool the model trained on
    # unseen: keys drawn from a disjoint pool (identical distribution)
    seen_p, seen_c, _ = make_fixed_eval(test_imgs, ccfg, cfg.n_keys,
                                        cfg.seed, cfg.eval_n)
    unseen_p, unseen_c, _ = make_fixed_eval(test_imgs, ccfg, cfg.n_keys,
                                            cfg.seed + UNSEEN_KEY_SEED_OFFSET,
                                            cfg.eval_n)
    seen_pt = to_tensor(seen_p)
    unseen_pt = to_tensor(unseen_p)

    floor = prior_floor(train_imgs, unseen_pt, "cpu")

    mkw = dict(cfg.model_kw)
    mkw.setdefault("in_ch", {"bits": 24, "both": 27, "pixel": 3}[cfg.input_repr]
                   + (2 if cfg.coords else 0) + (3 if cfg.prev_ch else 0))
    if cfg.out_mode == "categorical":
        mkw.setdefault("out_ch", 3 * 256)
        mkw.setdefault("logits", True)
    model = build_model(cfg.model, img_size=img_size, **mkw).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps))
        * (0.5 * (1 + np.cos(np.pi * min(1.0, s / cfg.steps)))))

    stream = PairStream(train_imgs, ccfg, cfg.n_keys, cfg.seed,
                        cfg.batch_size, steps_per_epoch=cfg.steps,
                        n_workers=cfg.workers).start()

    hist = []
    t0 = time.time()
    step = 0
    try:
        it = iter(stream)
        while step < cfg.steps:
            xc, xp = next(it)
            xc = encode_input(xc.to(device, non_blocking=True), cfg.input_repr,
                              cfg.coords, cfg.prev_ch)
            xp = xp.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                pred = model(xc)
                loss = (categorical_loss(pred, xp) if cfg.out_mode == "categorical"
                        else loss_fn(pred.float(), xp, cfg.loss))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % cfg.eval_every == 0 or step == cfg.steps:
                m_un = evaluate(infer(model, unseen_c, device, input_repr=cfg.input_repr, coords=cfg.coords, prev_ch=cfg.prev_ch,
                        out_mode=cfg.out_mode), unseen_pt)
                m_se = evaluate(infer(model, seen_c, device, input_repr=cfg.input_repr, coords=cfg.coords, prev_ch=cfg.prev_ch,
                        out_mode=cfg.out_mode), seen_pt)
                rec = {"step": step, "loss": loss.item(),
                       "seen": m_se, "unseen": m_un, "elapsed": time.time() - t0}
                hist.append(rec)
                if verbose:
                    print(f"[{cfg.name}] step {step:6d} loss {loss.item():.4f} | "
                          f"SEEN psnr {m_se['psnr']:6.2f} top1@100 {m_se.get('top1_pool100',0):.3f} | "
                          f"UNSEEN psnr {m_un['psnr']:6.2f} top1@100 {m_un.get('top1_pool100',0):.3f} | "
                          f"floor {floor['psnr']:.2f}", flush=True)
    finally:
        stream.stop()

    # ---- final evaluation with controls ---------------------------------
    pred_seen = infer(model, seen_c, device, input_repr=cfg.input_repr, coords=cfg.coords, prev_ch=cfg.prev_ch,
                        out_mode=cfg.out_mode)
    pred_unseen = infer(model, unseen_c, device, input_repr=cfg.input_repr, coords=cfg.coords, prev_ch=cfg.prev_ch,
                        out_mode=cfg.out_mode)
    m_seen = evaluate(pred_seen, seen_pt)
    m_unseen = evaluate(pred_unseen, unseen_pt)

    # Control 2: mismatched ciphertext -> quantifies hallucination from prior
    perm = np.random.default_rng(cfg.seed).permutation(len(unseen_c))
    m_mismatch = evaluate(infer(model, unseen_c[perm], device, input_repr=cfg.input_repr, coords=cfg.coords, prev_ch=cfg.prev_ch,
                        out_mode=cfg.out_mode), unseen_pt)

    result = {
        "config": asdict(cfg),
        "cipher_tag": ccfg.tag(),
        "n_params": n_par,
        "floor": floor,
        "seen": m_seen,
        "unseen": m_unseen,
        "mismatch_control": m_mismatch,
        "gain_over_floor_unseen": gain_over_floor(m_unseen, floor),
        "gain_over_floor_seen": gain_over_floor(m_seen, floor),
        "kgg_psnr": kgg(m_seen, m_unseen, floor, "psnr"),
        "kgg_top1_100": kgg(m_seen, m_unseen, floor, "top1_pool100")
        if "top1_pool100" in m_seen else None,
        "history": hist,
        "train_seconds": time.time() - t0,
    }

    out_dir = Path(out_dir or RUNS / cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    torch.save({"model": model.state_dict(), "config": asdict(cfg)},
               out_dir / "model.pt")
    np.savez_compressed(out_dir / "samples.npz",
                        plain=unseen_p[:32], cipher=unseen_c[:32],
                        recon=(pred_unseen[:32].numpy() * 255).astype(np.uint8))
    return result
