"""Parameterised chaos-based image cipher (Fridrich permutation-diffusion
architecture) plus an AES-CTR negative control behind the same interface.

The knobs are the independent variables of the whole study: each one removes
or adds a specific structural property, so attack success can be measured as a
function of cipher structure rather than as a single pass/fail.
"""
from dataclasses import dataclass, asdict, field

import numpy as np
from numba import njit, prange

from .maps import chaotic_stream, sample_keys, argsort_rows, KEY_BOUNDS

# Constant key used when perm_keyed=False, so the permutation is identical for
# every secret key (a deliberately key-independent structural component).
FIXED_PERM_KEY = np.array([[0.31415926535, 3.95, 0.05, 0.3]], dtype=np.float64)


@dataclass
class CipherConfig:
    name: str = "chaos"              # "chaos" | "aes"
    map: str = "logistic"            # logistic | tent | henon | pwlcm
    rounds: int = 2
    mode: str = "permute_diffuse"    # permute_only | diffuse_only | permute_diffuse
    perm_scope: str = "full2d"       # full2d | rowcol | blockwise
    block_size: int = 64             # flat block length when perm_scope=blockwise
    perm_keyed: bool = True          # False -> permutation shared across all keys
    feedback: str = "cbc"            # cbc | none
    precision: str = "float64"       # float64 | float32
    warmup: int = 500

    def tag(self):
        if self.name == "aes":
            return "aes"
        return (f"{self.map}_r{self.rounds}_{self.mode}_{self.perm_scope}"
                f"_{'keyed' if self.perm_keyed else 'fixedperm'}"
                f"_{self.feedback}_{self.precision}")

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# diffusion kernels
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _diffuse_fwd(p, k, iv, out):
    B, N = p.shape
    for b in prange(B):
        prev = iv[b]
        for i in range(N):
            c = (np.uint8((p[b, i] + prev) & 255)) ^ k[b, i]
            out[b, i] = c
            prev = c


@njit(cache=True, parallel=True)
def _diffuse_inv(c, k, iv, out):
    B, N = c.shape
    for b in prange(B):
        prev = iv[b]
        for i in range(N):
            cur = c[b, i]
            v = cur ^ k[b, i]
            out[b, i] = np.uint8((np.int64(v) - np.int64(prev)) & 255)
            prev = cur


# --------------------------------------------------------------------------
# permutation helpers
# --------------------------------------------------------------------------
def _apply_perm(flat, perm):
    return np.take_along_axis(flat, perm, axis=1)


def _apply_perm_inv(flat, perm):
    out = np.empty_like(flat)
    np.put_along_axis(out, perm, flat, axis=1)
    return out


def _perm_from_stream(s):
    """Sort-based permutation: argsort of a chaotic slice."""
    return argsort_rows(s)


class ChaosCipher:
    """Chaos-based image cipher with explicit structural knobs."""

    def __init__(self, cfg: CipherConfig):
        self.cfg = cfg

    # -- stream budgeting ---------------------------------------------------
    def _perm_len(self, H, W, C):
        s = self.cfg.perm_scope
        if s == "full2d":
            return H * W * C
        if s == "rowcol":
            return H + W
        if s == "blockwise":
            return H * W * C
        raise ValueError(s)

    def _budget(self, H, W, C):
        N = H * W * C
        per_round = 0
        if self.cfg.mode in ("permute_only", "permute_diffuse"):
            per_round += self._perm_len(H, W, C)
        if self.cfg.mode in ("diffuse_only", "permute_diffuse"):
            per_round += N
        return per_round * self.cfg.rounds + 1   # +1 value for the IV

    # -- keystream ----------------------------------------------------------
    def _streams(self, keys, H, W, C):
        cfg = self.cfg
        n = self._budget(H, W, C)
        s = chaotic_stream(cfg.map, keys, n, warmup=cfg.warmup,
                           precision=cfg.precision)
        if not cfg.perm_keyed:
            # permutation drawn from a constant key => identical for all keys
            fixed = np.repeat(FIXED_PERM_KEY, keys.shape[0], axis=0)
            s_fixed = chaotic_stream(cfg.map, fixed, n, warmup=cfg.warmup,
                                     precision=cfg.precision)
        else:
            s_fixed = s
        return s, s_fixed

    @staticmethod
    def _bytes_from(s, precision):
        # take mid-order bits of the mantissa; float32 has fewer to give,
        # which is precisely the degradation the precision knob models
        shift = 2 ** 32 if precision == "float64" else 2 ** 16
        return (np.floor(s * shift).astype(np.uint64) % 256).astype(np.uint8)

    # -- permutation dispatch ----------------------------------------------
    def _permute(self, x, s_slice, H, W, C, inverse=False):
        scope = self.cfg.perm_scope
        B = x.shape[0]
        if scope == "full2d":
            perm = _perm_from_stream(s_slice)
            flat = x.reshape(B, -1)
            out = _apply_perm_inv(flat, perm) if inverse else _apply_perm(flat, perm)
            return out.reshape(B, H, W, C)
        if scope == "rowcol":
            ph = _perm_from_stream(s_slice[:, :H])
            pw = _perm_from_stream(s_slice[:, H:H + W])
            if inverse:
                out = np.empty_like(x)
                np.put_along_axis(out, pw[:, None, :, None].repeat(H, 1).repeat(C, 3), x, axis=2)
                out2 = np.empty_like(out)
                np.put_along_axis(out2, ph[:, :, None, None].repeat(W, 2).repeat(C, 3), out, axis=1)
                return out2
            out = np.take_along_axis(x, ph[:, :, None, None].repeat(W, 2).repeat(C, 3), axis=1)
            out = np.take_along_axis(out, pw[:, None, :, None].repeat(H, 1).repeat(C, 3), axis=2)
            return out
        if scope == "blockwise":
            N = H * W * C
            bs = self.cfg.block_size
            assert N % bs == 0, f"block_size {bs} must divide N={N}"
            nb = N // bs
            flat = x.reshape(B, nb, bs)
            sb = s_slice[:, :N].reshape(B, nb, bs)
            perm = argsort_rows(sb.reshape(B*nb, bs)).reshape(B, nb, bs)
            if inverse:
                out = np.empty_like(flat)
                np.put_along_axis(out, perm, flat, axis=2)
            else:
                out = np.take_along_axis(flat, perm, axis=2)
            return out.reshape(B, H, W, C)
        raise ValueError(scope)

    # -- public API ---------------------------------------------------------
    def encrypt(self, imgs, keys):
        """imgs: (B,H,W,C) uint8, keys: (B,4) float64 -> (B,H,W,C) uint8"""
        cfg = self.cfg
        B, H, W, C = imgs.shape
        N = H * W * C
        s, s_perm_src = self._streams(keys, H, W, C)
        iv = self._bytes_from(s[:, -1:], cfg.precision)[:, 0]

        x = imgs.copy()
        pos = 0
        plen = self._perm_len(H, W, C)
        for _ in range(cfg.rounds):
            if cfg.mode in ("permute_only", "permute_diffuse"):
                x = self._permute(x, s_perm_src[:, pos:pos + plen], H, W, C, inverse=False)
                pos += plen
            if cfg.mode in ("diffuse_only", "permute_diffuse"):
                k = self._bytes_from(s[:, pos:pos + N], cfg.precision)
                pos += N
                flat = np.ascontiguousarray(x.reshape(B, N))
                if cfg.feedback == "cbc":
                    out = np.empty_like(flat)
                    _diffuse_fwd(flat, k, iv, out)
                else:
                    out = flat ^ k
                x = out.reshape(B, H, W, C)
        return x

    def decrypt(self, ciph, keys):
        cfg = self.cfg
        B, H, W, C = ciph.shape
        N = H * W * C
        s, s_perm_src = self._streams(keys, H, W, C)
        iv = self._bytes_from(s[:, -1:], cfg.precision)[:, 0]

        plen = self._perm_len(H, W, C)
        # replay forward offsets so rounds can be undone in reverse
        offsets = []
        pos = 0
        for _ in range(cfg.rounds):
            po = do = None
            if cfg.mode in ("permute_only", "permute_diffuse"):
                po = pos
                pos += plen
            if cfg.mode in ("diffuse_only", "permute_diffuse"):
                do = pos
                pos += N
            offsets.append((po, do))

        x = ciph.copy()
        for po, do in reversed(offsets):
            if do is not None:
                k = self._bytes_from(s[:, do:do + N], cfg.precision)
                flat = np.ascontiguousarray(x.reshape(B, N))
                if cfg.feedback == "cbc":
                    out = np.empty_like(flat)
                    _diffuse_inv(flat, k, iv, out)
                else:
                    out = flat ^ k
                x = out.reshape(B, H, W, C)
            if po is not None:
                x = self._permute(x, s_perm_src[:, po:po + plen], H, W, C, inverse=True)
        return x


# --------------------------------------------------------------------------
# AES-CTR negative control
# --------------------------------------------------------------------------
class AESCipher:
    """AES-128-CTR over the flattened image. The attack MUST fail on this;
    if it does not, the pipeline is leaking plaintext somewhere."""

    def __init__(self, cfg: CipherConfig = None):
        self.cfg = cfg or CipherConfig(name="aes")

    @staticmethod
    def _key_bytes(k):
        return np.ascontiguousarray(k, dtype=np.float64).tobytes()[:16]

    def encrypt(self, imgs, keys):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        B, H, W, C = imgs.shape
        flat = imgs.reshape(B, -1)
        out = np.empty_like(flat)
        for b in range(B):
            enc = Cipher(algorithms.AES(self._key_bytes(keys[b])),
                         modes.CTR(b"\x00" * 16)).encryptor()
            out[b] = np.frombuffer(enc.update(flat[b].tobytes()) + enc.finalize(),
                                   dtype=np.uint8)
        return out.reshape(B, H, W, C)

    def decrypt(self, ciph, keys):
        return self.encrypt(ciph, keys)   # CTR is its own inverse


def build_cipher(cfg: CipherConfig):
    return AESCipher(cfg) if cfg.name == "aes" else ChaosCipher(cfg)


def sample_cipher_keys(cfg: CipherConfig, n, rng):
    map_name = "logistic" if cfg.name == "aes" else cfg.map
    return sample_keys(map_name, n, rng)
