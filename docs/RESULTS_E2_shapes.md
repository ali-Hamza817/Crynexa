# E2 — cipher-strength sweep, structured plaintexts (`shapes`)

60/60 runs. 15 cipher configurations × {same-key ceiling, key-agnostic} ×
{ResUNet, LinearMixer}. 5 000 steps each, batch 256, L1 loss.

Chance for top-1 @ pool 100 is **0.010**. "Same-key ceiling" is the seen-key
retrieval of the `n_keys=1` run: the best this attacker achieves when the key is
fixed. It is the reference every key-agnostic number must be read against.

---

## Results

| cipher config | attacker | same-key ceiling | unseen-key | mismatch | gain dB | verdict |
|---|---|---|---|---|---|---|
| posctrl_fixedperm_only | linearmixer | 1.000 | **1.000** | 0.010 | +15.50 | control ✓ |
| posctrl_fixedperm_only | resunet | 1.000 | **1.000** | 0.010 | +13.86 | control ✓ |
| **scope_rowcol_permonly** | **linearmixer** | 1.000 | **0.100** | 0.010 | **+3.67** | **BROKEN** |
| **scope_rowcol_permonly** | **resunet** | 1.000 | **0.096** | 0.011 | **+3.15** | **BROKEN** |
| permute_only_r1 | resunet | 1.000 | 0.020 | 0.010 | +0.49 | marginal |
| permute_only_r2 | resunet | 1.000 | 0.017 | 0.011 | +0.49 | marginal |
| permute_only_r1 | linearmixer | 1.000 | 0.014 | 0.009 | +0.86 | marginal |
| permute_only_r2 | linearmixer | 1.000 | 0.014 | 0.008 | +0.85 | marginal |
| no_feedback | linearmixer | 1.000 | 0.013 | 0.012 | −0.00 | resisted |
| no_feedback | resunet | 0.353 | 0.010 | 0.011 | −0.00 | resisted |
| negctrl_aes | linearmixer | 1.000 | 0.009 | 0.008 | −0.00 | control ✓ |
| negctrl_aes | resunet | 1.000 | 0.010 | 0.010 | −0.00 | control ✓ |
| rounds1 / 2 / 3 / 4 | both | **~0.010** | ~0.011 | ~0.010 | −0.00 | **no information** |
| diffuse_only_r1 | both | **~0.010** | ~0.011 | ~0.010 | −0.00 | **no information** |
| scope_rowcol | both | **~0.011** | ~0.010 | ~0.011 | −0.00 | **no information** |
| scope_blockwise64 | both | **~0.011** | ~0.010 | ~0.011 | −0.00 | **no information** |
| fixedperm_keyed_diff | both | **~0.010** | ~0.011 | ~0.010 | −0.00 | **no information** |
| float32 | both | **~0.010** | ~0.010 | ~0.011 | −0.00 | **no information** |

---

## 1. The instrument is calibrated

The two controls bracket the full range, and both have a same-key ceiling of
1.000, so attacker capability is not in question in either case. The only
variable separating them is key dependence:

- **No key dependence** (fixed permutation): unseen-key 1.000, +15.5 dB
- **Full key dependence** (AES-CTR): unseen-key 0.009, −0.00 dB

A ~100× separation. Measurements landing between these poles are meaningful.

## 2. Row/column permutation is broken without the key

The headline positive result. `scope_rowcol_permonly` permutes row order and
column order independently rather than scrambling pixels freely. That preserves
the **multiset of whole rows and whole columns under every key** — key-independent
structure by construction.

Both attacker families find it independently (0.100 and 0.096, i.e. **10× chance**,
+3.67 and +3.15 dB) with the mismatch control at chance (0.010, 0.011), so the
gain is not hallucination from the plaintext prior.

Full 2-D permutation of the same cipher leaks far less (0.014–0.020, 1.4–2×
chance). The contrast localises the weakness precisely: **it is the restricted
permutation scope, not permutation per se, that is attackable key-agnostically.**
Many published chaos schemes permute rows and columns separately because it is
cheaper than a full scramble; this measures what that costs.

## 3. A keyed permutation plus keyed diffusion resisted

`no_feedback` (keyed full-2D permutation + keyed XOR, no CBC chain) has a valid
same-key ceiling of 1.000 for LinearMixer and 0.353 for ResUNet, yet drops to
0.013 / 0.010 — chance — under unlimited key diversity. Because the ceiling is
high, this is a genuine resistance result rather than an attacker failure: the
attacker demonstrably *can* invert this cipher when the key is fixed, and
demonstrably learns nothing key-independent from it.

This is the clean negative that the key-agnostic hypothesis predicts (§2 of the
analysis): the chaotic key→keystream map is non-smooth, so nothing interpolates
across key space.

## 4. Twelve of fifteen configurations returned NO security information

Every configuration using CBC feedback diffusion — `rounds1–4`, `diffuse_only`,
`scope_rowcol`, `scope_blockwise64`, `fixedperm_keyed_diff`, `float32` — has a
**same-key ceiling at chance**. The attacker cannot invert them even with the key
fixed and constant across every training example.

Their unseen-key numbers must therefore **not** be reported as resistance. "The
cipher resists key-independent attack" and "this attacker cannot invert this
cipher at all" produce identical numbers, and only the ceiling distinguishes
them. This is finding F7 in practice, and it affects 80% of the sweep.

The pattern is unambiguous:

| feedback | same-key ceiling |
|---|---|
| none (or permutation-only) | 1.000 |
| CBC chain | ~0.010 (chance) |

The CBC inverse is `p_i = ((c_i XOR k_i) - c_{i-1}) mod 256`. The flat predecessor
`c_{i-1}` of channel 0 at `(h,w)` is channel `C-1` at `(h,w-1)`, an alignment a
convolution kernel cannot cheaply construct, and in bit-planes the mod-256
subtraction is an 8-deep borrow chain.

**Action taken.** A flat-raster predecessor channel was added to the attacker
input (`prev_ch`). This supplies algorithm structure — the scan order — not key
material, consistent with Kerckhoffs's principle. Whether it lifts the CBC
ceiling off chance is being measured; the CIFAR-10 sweep uses whichever setting
is demonstrated to give the attacker a real ceiling, so that its cells are
interpretable.

Until that ceiling is raised, **no security claim is made for any CBC
configuration.**

---

## Honest summary

| claim | supported? |
|---|---|
| Instrument detects a key-independent leak | yes — positive control 1.000, +15.5 dB |
| Instrument does not hallucinate | yes — AES and all mismatch controls at chance |
| Row/column permutation leaks key-agnostically | **yes — 10× chance, both attackers, mismatch clean** |
| Full 2-D permutation leaks much less | yes — 1.4–2× chance |
| Permutation scope governs key-agnostic recoverability | yes, for permutation-only ciphers |
| Keyed permutation + keyed XOR resists | yes — ceiling 1.000, key-agnostic at chance |
| Round count governs resistance | **not measured** — ceiling at chance |
| Keystream precision governs resistance | **not measured** — ceiling at chance |
| CBC feedback confers security | **not shown** — the attacker never reached its ceiling |
