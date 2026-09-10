"""Standard cipher-characterisation statistics.

These describe the CIPHER, not the attack. They exist to prove the implemented
scheme meets published chaos-encryption standards and is therefore not a
strawman. Reference targets for a sound scheme:
    entropy              -> 7.99+ (8.0 ideal for 8-bit)
    adjacent correlation -> ~0.00
    NPCR                 -> ~99.61 %
    UACI                 -> ~33.46 %
"""
import numpy as np


def shannon_entropy(imgs):
    """Mean per-image entropy of the 8-bit value distribution."""
    out = []
    for im in imgs:
        h = np.bincount(im.ravel(), minlength=256).astype(np.float64)
        p = h / h.sum()
        p = p[p > 0]
        out.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(out))


def adjacent_correlation(imgs, direction="h", n_pairs=4000, rng=None):
    """Correlation coefficient between randomly sampled adjacent pixel pairs."""
    rng = rng or np.random.default_rng(0)
    xs, ys = [], []
    B, H, W, C = imgs.shape
    for _ in range(n_pairs):
        b = rng.integers(B); c = rng.integers(C)
        if direction == "h":
            i = rng.integers(H); j = rng.integers(W - 1)
            xs.append(imgs[b, i, j, c]); ys.append(imgs[b, i, j + 1, c])
        elif direction == "v":
            i = rng.integers(H - 1); j = rng.integers(W)
            xs.append(imgs[b, i, j, c]); ys.append(imgs[b, i + 1, j, c])
        else:
            i = rng.integers(H - 1); j = rng.integers(W - 1)
            xs.append(imgs[b, i, j, c]); ys.append(imgs[b, i + 1, j + 1, c])
    xs = np.array(xs, np.float64); ys = np.array(ys, np.float64)
    if xs.std() == 0 or ys.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def npcr_uaci(c1, c2):
    """Plaintext-sensitivity: two ciphertexts from plaintexts differing in one bit."""
    d = (c1 != c2).astype(np.float64)
    npcr = 100.0 * d.mean()
    uaci = 100.0 * (np.abs(c1.astype(np.float64) - c2.astype(np.float64)) / 255.0).mean()
    return float(npcr), float(uaci)


def key_sensitivity(cipher, imgs, keys, eps=1e-12):
    """Ciphertext difference when the key is perturbed by eps."""
    k2 = keys.copy()
    k2[:, 0] += eps
    c1 = cipher.encrypt(imgs, keys)
    c2 = cipher.encrypt(imgs, k2)
    return npcr_uaci(c1, c2)


def characterise(cipher, imgs, keys, rng=None):
    """Full cipher-characterisation report."""
    rng = rng or np.random.default_rng(0)
    ct = cipher.encrypt(imgs, keys)

    # plaintext sensitivity: flip the LSB of one pixel in each image
    p2 = imgs.copy()
    p2[:, 0, 0, 0] ^= 1
    ct2 = cipher.encrypt(p2, keys)
    npcr, uaci = npcr_uaci(ct, ct2)
    k_npcr, k_uaci = key_sensitivity(cipher, imgs, keys)

    return {
        "entropy_plain": shannon_entropy(imgs),
        "entropy_cipher": shannon_entropy(ct),
        "corr_h_plain": adjacent_correlation(imgs, "h", rng=rng),
        "corr_h_cipher": adjacent_correlation(ct, "h", rng=rng),
        "corr_v_cipher": adjacent_correlation(ct, "v", rng=rng),
        "corr_d_cipher": adjacent_correlation(ct, "d", rng=rng),
        "npcr": npcr,
        "uaci": uaci,
        "key_sens_npcr": k_npcr,
        "key_sens_uaci": k_uaci,
    }
