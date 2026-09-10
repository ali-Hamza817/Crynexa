"""Runnable checks for the cipher and the experiment controls.

    python tests/test_cipher.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crynexa.chaos.cipher import CipherConfig, build_cipher, sample_cipher_keys
from crynexa.data.pairs import KeyPool
from crynexa.metrics.cipher_stats import characterise


def test_roundtrip_all_configs():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    n = 0
    for m, mode, scope, fb, pk, r, prec in itertools.product(
            ["logistic", "tent", "henon", "pwlcm"],
            ["permute_only", "diffuse_only", "permute_diffuse"],
            ["full2d", "rowcol", "blockwise"], ["cbc", "none"],
            [True, False], [1, 3], ["float64", "float32"]):
        cfg = CipherConfig(map=m, mode=mode, perm_scope=scope, feedback=fb,
                           perm_keyed=pk, rounds=r, precision=prec)
        c = build_cipher(cfg)
        keys = sample_cipher_keys(cfg, 4, rng)
        assert np.array_equal(c.decrypt(c.encrypt(imgs, keys), keys), imgs), cfg.tag()
        n += 1
    print(f"  roundtrip exact for {n} configs")


def test_positive_control_is_key_independent():
    """perm_keyed=False + permute_only must give an IDENTICAL transform for
    every key. This is the positive control: an attack that cannot break this
    cannot detect any leak, so the whole instrument would be uncalibrated."""
    rng = np.random.default_rng(1)
    imgs = rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    cfg = CipherConfig(mode="permute_only", perm_keyed=False, rounds=1)
    c = build_cipher(cfg)
    k1 = sample_cipher_keys(cfg, 4, rng)
    k2 = sample_cipher_keys(cfg, 4, rng)
    assert not np.allclose(k1, k2), "keys must actually differ"
    assert np.array_equal(c.encrypt(imgs, k1), c.encrypt(imgs, k2)), \
        "positive control is not key-independent"
    print("  positive control: ciphertext identical under different keys")


def test_keyed_cipher_is_key_dependent():
    rng = np.random.default_rng(2)
    imgs = rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    for cfg in [CipherConfig(), CipherConfig(name="aes")]:
        c = build_cipher(cfg)
        k1 = sample_cipher_keys(cfg, 4, rng)
        k2 = sample_cipher_keys(cfg, 4, rng)
        a, b = c.encrypt(imgs, k1), c.encrypt(imgs, k2)
        frac = (a != b).mean()
        assert frac > 0.9, f"{cfg.tag()} barely key-dependent ({frac:.3f})"
        print(f"  {cfg.tag()[:38]:38s} differs in {frac*100:.2f}% of bytes across keys")


def test_permutation_preserves_histogram():
    """A permutation-only cipher preserves the pixel multiset exactly, for
    every key. That is key-independent structure and is why such schemes are
    attackable without the key."""
    rng = np.random.default_rng(3)
    imgs = rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    cfg = CipherConfig(mode="permute_only", rounds=2)
    c = build_cipher(cfg)
    ct = c.encrypt(imgs, sample_cipher_keys(cfg, 4, rng))
    for i in range(len(imgs)):
        assert np.array_equal(np.bincount(imgs[i].ravel(), minlength=256),
                              np.bincount(ct[i].ravel(), minlength=256))
    print("  permute_only preserves the histogram exactly (key-independent leak)")


def test_key_pool_sharing():
    """All producer threads must share one pool, or n_keys is meaningless."""
    cfg = CipherConfig()
    pools = [KeyPool(cfg, 4, 1000 * w, pool_seed=0).pool for w in range(6)]
    assert all(np.array_equal(pools[0], p) for p in pools)
    assert len({tuple(KeyPool(cfg, 1, 1000 * w, pool_seed=0).pool[0])
                for w in range(6)}) == 1
    seen = KeyPool(cfg, 8, 0).pool
    unseen = KeyPool(cfg, 8, 999983).pool
    assert not any(np.allclose(a, b) for a in seen for b in unseen)
    print("  key pools shared across workers; seen/unseen disjoint")


def test_cipher_meets_published_standards():
    """The reference configuration must match published chaos-cipher targets,
    or it is a strawman and every attack result on it is worthless."""
    rng = np.random.default_rng(4)
    H = W = 32
    gx, gy = np.meshgrid(np.linspace(0, 255, W), np.linspace(0, 255, H))
    base = np.stack([np.stack([gx, gy, (gx + gy) / 2], -1).astype(np.uint8)] * 32)
    cfg = CipherConfig(rounds=2)
    st = characterise(build_cipher(cfg), base, sample_cipher_keys(cfg, 32, rng),
                      rng=np.random.default_rng(5))
    # 7.9407 is the finite-sample ceiling for 3072 pixels over 256 bins
    assert st["entropy_cipher"] > 7.93, st["entropy_cipher"]
    assert abs(st["corr_h_cipher"]) < 0.06, st["corr_h_cipher"]
    assert 99.0 < st["npcr"] < 100.0, st["npcr"]
    assert 32.0 < st["uaci"] < 35.0, st["uaci"]
    print(f"  entropy {st['entropy_cipher']:.3f} (ceiling 7.9407)  "
          f"corr {st['corr_h_cipher']:+.4f}  "
          f"NPCR {st['npcr']:.3f}%  UACI {st['uaci']:.3f}%")


if __name__ == "__main__":
    for fn in [test_roundtrip_all_configs, test_positive_control_is_key_independent,
               test_keyed_cipher_is_key_dependent, test_permutation_preserves_histogram,
               test_key_pool_sharing, test_cipher_meets_published_standards]:
        print(f"{fn.__name__}:")
        fn()
    print("\nall checks passed")
