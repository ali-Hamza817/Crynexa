"""Batched chaotic map stream generators.

Each image in a batch is encrypted under its own key, so trajectories for
different batch elements are independent and can be generated in parallel.
Iteration within one trajectory is inherently sequential.

All maps emit values in [0, 1).
"""
import numpy as np
from numba import njit, prange

MAP_IDS = {"logistic": 0, "tent": 1, "henon": 2, "pwlcm": 3}
MAP_NAMES = {v: k for k, v in MAP_IDS.items()}

# Key-space bounds, sampled uniformly. Ranges chosen to stay inside the
# chaotic regime of each map.
KEY_BOUNDS = {
    "logistic": [(0.1, 0.9), (3.90, 4.00), (0.0, 0.0), (0.0, 0.0)],   # x0, r
    "tent":     [(0.1, 0.9), (1.90, 2.00), (0.0, 0.0), (0.0, 0.0)],   # x0, mu
    "henon":    [(-0.1, 0.1), (1.35, 1.42), (-0.1, 0.1), (0.3, 0.3)],  # x0, a, y0, b
    "pwlcm":    [(0.1, 0.9), (0.10, 0.49), (0.0, 0.0), (0.0, 0.0)],   # x0, p
}


@njit(cache=True, parallel=True, fastmath=False)
def _stream_f64(map_id, params, warmup, out):
    B, n = out.shape
    for b in prange(B):
        if map_id == 0:      # logistic
            x = params[b, 0]
            r = params[b, 1]
            for _ in range(warmup):
                x = r * x * (1.0 - x)
            for i in range(n):
                x = r * x * (1.0 - x)
                out[b, i] = x
        elif map_id == 1:    # tent
            x = params[b, 0]
            mu = params[b, 1]
            for _ in range(warmup):
                x = mu * x if x < 0.5 else mu * (1.0 - x)
            for i in range(n):
                x = mu * x if x < 0.5 else mu * (1.0 - x)
                out[b, i] = x
        elif map_id == 2:    # henon (x normalised from [-1.5, 1.5] to [0, 1))
            x = params[b, 0]
            a = params[b, 1]
            y = params[b, 2]
            bb = params[b, 3]
            for _ in range(warmup):
                xn = 1.0 - a * x * x + y
                y = bb * x
                x = xn
            for i in range(n):
                xn = 1.0 - a * x * x + y
                y = bb * x
                x = xn
                v = (x + 1.5) / 3.0
                if v < 0.0:
                    v = 0.0
                elif v >= 1.0:
                    v = 0.999999999
                out[b, i] = v
        else:                # pwlcm
            x = params[b, 0]
            p = params[b, 1]
            for _ in range(warmup):
                if x < p:
                    x = x / p
                elif x < 0.5:
                    x = (x - p) / (0.5 - p)
                elif x < 1.0 - p:
                    x = (1.0 - p - x) / (0.5 - p)
                else:
                    x = (1.0 - x) / p
            for i in range(n):
                if x < p:
                    x = x / p
                elif x < 0.5:
                    x = (x - p) / (0.5 - p)
                elif x < 1.0 - p:
                    x = (1.0 - p - x) / (0.5 - p)
                else:
                    x = (1.0 - x) / p
                if x < 0.0:
                    x = 0.0
                elif x >= 1.0:
                    x = 0.999999999
                out[b, i] = x


@njit(cache=True, parallel=True, fastmath=False)
def _stream_f32(map_id, params, warmup, out):
    """float32 trajectory — models finite-precision keystream degradation."""
    B, n = out.shape
    for b in prange(B):
        if map_id == 0:
            x = np.float32(params[b, 0])
            r = np.float32(params[b, 1])
            for _ in range(warmup):
                x = r * x * (np.float32(1.0) - x)
            for i in range(n):
                x = r * x * (np.float32(1.0) - x)
                out[b, i] = x
        elif map_id == 1:
            x = np.float32(params[b, 0])
            mu = np.float32(params[b, 1])
            for _ in range(warmup):
                x = mu * x if x < np.float32(0.5) else mu * (np.float32(1.0) - x)
            for i in range(n):
                x = mu * x if x < np.float32(0.5) else mu * (np.float32(1.0) - x)
                out[b, i] = x
        elif map_id == 2:
            x = np.float32(params[b, 0])
            a = np.float32(params[b, 1])
            y = np.float32(params[b, 2])
            bb = np.float32(params[b, 3])
            for _ in range(warmup):
                xn = np.float32(1.0) - a * x * x + y
                y = bb * x
                x = xn
            for i in range(n):
                xn = np.float32(1.0) - a * x * x + y
                y = bb * x
                x = xn
                v = (x + np.float32(1.5)) / np.float32(3.0)
                if v < np.float32(0.0):
                    v = np.float32(0.0)
                elif v >= np.float32(1.0):
                    v = np.float32(0.99999)
                out[b, i] = v
        else:
            x = np.float32(params[b, 0])
            p = np.float32(params[b, 1])
            half = np.float32(0.5)
            one = np.float32(1.0)
            for _ in range(warmup):
                if x < p:
                    x = x / p
                elif x < half:
                    x = (x - p) / (half - p)
                elif x < one - p:
                    x = (one - p - x) / (half - p)
                else:
                    x = (one - x) / p
            for i in range(n):
                if x < p:
                    x = x / p
                elif x < half:
                    x = (x - p) / (half - p)
                elif x < one - p:
                    x = (one - p - x) / (half - p)
                else:
                    x = (one - x) / p
                if x < np.float32(0.0):
                    x = np.float32(0.0)
                elif x >= one:
                    x = np.float32(0.99999)
                out[b, i] = x


def chaotic_stream(map_name, params, n, warmup=500, precision="float64"):
    """Generate `n` chaotic values per batch element.

    params : (B, 4) float64 key parameters
    returns: (B, n) float64 in [0, 1)
    """
    map_id = MAP_IDS[map_name]
    params = np.ascontiguousarray(params, dtype=np.float64)
    B = params.shape[0]
    if precision == "float32":
        out = np.empty((B, n), dtype=np.float32)
        _stream_f32(map_id, params, warmup, out)
        return out.astype(np.float64)
    out = np.empty((B, n), dtype=np.float64)
    _stream_f64(map_id, params, warmup, out)
    return out


def sample_keys(map_name, n, rng):
    """Sample `n` keys uniformly from the full key space of `map_name`."""
    bounds = KEY_BOUNDS[map_name]
    keys = np.empty((n, 4), dtype=np.float64)
    for j, (lo, hi) in enumerate(bounds):
        keys[:, j] = lo if hi == lo else rng.uniform(lo, hi, size=n)
    return keys


@njit(cache=True, parallel=True)
def _argsort_rows(s, out):
    B = s.shape[0]
    for b in prange(B):
        out[b] = np.argsort(s[b])


def argsort_rows(s):
    """Row-wise argsort, parallel over the batch (np.argsort is single-threaded
    and is otherwise the dominant cost of permutation generation)."""
    out = np.empty(s.shape, dtype=np.int64)
    _argsort_rows(np.ascontiguousarray(s), out)
    return out
