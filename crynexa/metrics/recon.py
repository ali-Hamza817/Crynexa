"""Attack-success metrics.

Retrieval accuracy is the PRIMARY metric. PSNR/SSIM reward outputs that are
plausible; only retrieval shows that *this* plaintext was recovered rather than
*a* plausible image produced from the dataset prior.

Similarity is cosine on mean-centred images, so a constant (prior-only)
prediction becomes the zero vector, ties everywhere, and scores exactly at
chance -- which is the behaviour a sound metric must have.
"""
import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
    return (10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-12)))


def _gauss_win(ws, sigma, ch, device, dtype):
    g = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w2 = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return w2.expand(ch, 1, ws, ws).contiguous()


def ssim(pred, target, ws=11, sigma=1.5, max_val=1.0):
    ch = pred.shape[1]
    ws = min(ws, pred.shape[-1] if pred.shape[-1] % 2 == 1 else pred.shape[-1] - 1)
    w = _gauss_win(ws, sigma, ch, pred.device, pred.dtype)
    pad = ws // 2
    mu1 = F.conv2d(pred, w, padding=pad, groups=ch)
    mu2 = F.conv2d(target, w, padding=pad, groups=ch)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred * pred, w, padding=pad, groups=ch) - mu1s
    s2 = F.conv2d(target * target, w, padding=pad, groups=ch) - mu2s
    s12 = F.conv2d(pred * target, w, padding=pad, groups=ch) - mu12
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return m.flatten(1).mean(1)


def _centred_unit(x):
    v = x.flatten(1)
    v = v - v.mean(1, keepdim=True)
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def retrieval_accuracy(pred, target, pool_sizes=(10, 100, 1000), repeats=20, seed=0):
    """Top-1 accuracy: is the true plaintext the nearest candidate?

    For each query a pool of `K-1` distractors plus the true plaintext is drawn
    from the evaluation set. Chance is 1/K.
    """
    a = _centred_unit(pred)
    b = _centred_unit(target)
    n = a.shape[0]
    sim = a @ b.t()                                   # (n, n)
    # break exact ties (prior-only predictions) uniformly at random
    g = torch.Generator(device="cpu").manual_seed(seed)
    sim = sim + torch.rand(sim.shape, generator=g).to(sim.device) * 1e-6
    diag = sim.diag()
    out = {}
    for K in pool_sizes:
        if K > n:
            continue
        hits = 0
        total = 0
        rows = torch.arange(n, device=sim.device).unsqueeze(1)
        for r in range(repeats):
            idx = torch.randint(0, n, (n, K - 1), generator=g).to(sim.device)
            distract = torch.gather(sim, 1, idx)
            # the query's own target must never appear as its own distractor,
            # or accuracy is capped at 1-(1-1/n)^(K-1) instead of 1.0
            distract = distract.masked_fill(idx == rows, float("-inf"))
            hits += (diag.unsqueeze(1) > distract).all(1).sum().item()
            total += n
        out[f"top1_pool{K}"] = hits / total
        out[f"chance_pool{K}"] = 1.0 / K
    return out


@torch.no_grad()
def evaluate(pred, target, pool_sizes=(10, 100, 1000), seed=0):
    m = {
        "psnr": psnr(pred, target).mean().item(),
        "ssim": ssim(pred, target).mean().item(),
        "mse": F.mse_loss(pred, target).item(),
        "mae": F.l1_loss(pred, target).item(),
    }
    m.update(retrieval_accuracy(pred, target, pool_sizes, seed=seed))
    return m


def prior_floor(train_images_u8, target, device, pool_sizes=(10, 100, 1000)):
    """Control 1: metrics for the constant predictor P_hat = mean training image.

    This is the value corresponding to ZERO information extracted. Every attack
    number must be read against it.
    """
    mean_img = torch.from_numpy(
        train_images_u8.mean(axis=0).astype(np.float32) / 255.0
    ).permute(2, 0, 1).to(device)
    pred = mean_img.unsqueeze(0).expand_as(target).contiguous()
    return evaluate(pred, target, pool_sizes)


def gain_over_floor(metrics, floor):
    """Report attack numbers as gain over the zero-information floor."""
    return {
        "psnr_gain_db": metrics["psnr"] - floor["psnr"],
        "ssim_gain": metrics["ssim"] - floor["ssim"],
        **{f"{k}_gain": metrics[k] - floor[k]
           for k in metrics if k.startswith("top1_")},
    }


def kgg(seen, unseen, floor, key="psnr"):
    """Normalised Key Generalization Gap.

    0 -> attack transfers perfectly to unseen keys (full key-agnostic break)
    1 -> attack retains nothing on unseen keys (key mixing sound)
    """
    denom = seen[key] - floor[key]
    if abs(denom) < 1e-9:
        return float("nan")
    return float((seen[key] - unseen[key]) / denom)
