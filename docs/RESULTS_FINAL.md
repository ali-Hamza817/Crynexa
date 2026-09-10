# Final results — key-agnostic neural cryptanalysis of chaos-based image encryption

**223 trained models.** All numbers measured; none estimated or simulated.
Chance for top-1 retrieval @ pool 100 is **0.010**.

Baseline cipher: logistic map, 2 rounds, keyed full-2D permutation + keyed XOR
diffusion, float64. CBC feedback is a variant under test, not the baseline (§6).

"Ceiling" = seen-key retrieval of the `n_keys=1` run for that exact
configuration: what this attacker achieves when the key is fixed. A
configuration whose ceiling is at chance yields **no** security information,
because "the cipher resists" and "this attacker cannot invert it at all" then
produce identical numbers.

---

## 1. Controls — the instrument is calibrated on both datasets

| control | dataset | attacker | ceiling | unseen-key | mismatch | gain |
|---|---|---|---|---|---|---|
| key-independent permutation | CIFAR-10 | linearmixer | 1.000 | **1.000** | 0.010 | +20.42 dB |
| key-independent permutation | CIFAR-10 | resunet | 1.000 | **1.000** | 0.010 | +15.61 dB |
| key-independent permutation | shapes | linearmixer | 1.000 | **1.000** | 0.010 | +15.90 dB |
| AES-128-CTR | CIFAR-10 | both | 1.000 | **0.010** | 0.010 | +0.01 dB |
| AES-128-CTR | shapes | both | 1.000 | **0.011** | 0.009 | -0.00 dB |

Both controls have a ceiling of 1.000, so attacker capability is never in
question. The separation between them is ~100x. Every mismatch control sits at
chance, so no result below is hallucination from the plaintext prior.

---

## 2. Headline finding — row/column permutation is broken without the key

| dataset | attacker | ceiling | **unseen-key** | mismatch | gain |
|---|---|---|---|---|---|
| CIFAR-10 | **resunet** | 1.000 | **0.221** (22x chance) | 0.013 | **+2.63 dB** |
| CIFAR-10 | linearmixer | 1.000 | 0.087 (9x chance) | 0.011 | +2.07 dB |
| shapes | linearmixer | 1.000 | 0.100 (10x chance) | 0.010 | +3.67 dB |
| shapes | resunet | 1.000 | 0.096 (10x chance) | 0.011 | +3.14 dB |

Replicates across **two datasets and two attacker families**, with clean
mismatch controls throughout. Row/column scrambling permutes row order and
column order independently, preserving the multiset of whole rows and whole
columns **under every key** — key-independent structure by construction.

Full 2-D permutation of the same cipher leaks far less:

| config | CIFAR-10 unseen | shapes unseen |
|---|---|---|
| row/col permutation | **0.221 / 0.087** | 0.100 / 0.096 |
| full 2-D permutation | 0.045 / 0.025 | 0.018 / 0.013 |

So it is the **restricted permutation scope**, not permutation itself, that is
attackable without the key. Many published chaos schemes permute rows and
columns separately because it is cheaper than a full scramble; this quantifies
the cost of that shortcut.

The attacker ordering inverts between datasets: on CIFAR-10 the ResUNet wins
(0.221 vs 0.087) because natural-image statistics reward a convolutional prior
when reassembling displaced rows; on the synthetic set the two are level.
Reporting only one architecture would have understated this leak by 2.5x.

---

## 3. Mechanism — the diffusion stage carries all of the key-agnostic security

The decisive comparison. Both configurations use a **fully key-independent
permutation**; they differ only in whether a keyed diffusion stage follows:

| configuration | keyed diffusion? | ceiling | unseen-key | verdict |
|---|---|---|---|---|
| `posctrl_fixedperm_only` | no | 1.000 | **1.000** | completely broken |
| `fixedperm_keyed_diff` | yes | 1.000 | **0.010-0.017** | at chance |

A single keyed diffusion stage takes the attack from total recovery to chance,
**even though the permutation is identical for every key**. The same masking
applies to the row/column leak: `scope_rowcol` (row/col permutation *plus*
keyed diffusion) measures 0.010-0.013, versus 0.087-0.221 without it.

**Permutation contributes essentially nothing to key-agnostic security.
Diffusion contributes all of it.** For a scheme designer this is the actionable
result: the permutation stage may be chosen for speed, but the diffusion stage
must be keyed and must cover the whole image.

---

## 4. Axes that turned out not to matter

All of the following have a valid ceiling of ~1.000 and sit at exactly chance
under unlimited key diversity, on both datasets:

| axis | values tested | unseen-key |
|---|---|---|
| rounds | 1, 2, 3, 4 | 0.010-0.011 |
| permutation scope (with diffusion) | full2d, rowcol, blockwise-64 | 0.010-0.013 |
| keystream precision | float64, float32 | 0.009-0.010 |
| diffusion alone | diffuse_only, 1 round | 0.010-0.012 |

Two are worth stating explicitly because the literature treats them as security
parameters:

- **Round count buys nothing here.** One round of keyed permutation + keyed
  diffusion already reaches chance; rounds 2-4 do not improve on it. Against
  *this* attack class the marginal value of extra rounds is zero.
- **float32 keystream degradation does not create an exploitable leak.**
  Finite-precision chaos is a commonly cited weakness; it does not manifest as
  key-independent structure a neural attacker can use.

---

## 5. The key-diversity curve — capacity, not generalisation

Baseline cipher, CIFAR-10. `seen` is retrieval on keys **drawn from the training
pool**; `unseen` on disjoint keys.

| distinct training keys | seen (linearmixer) | seen (resunet) | **unseen** |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 0.008 / 0.007 |
| 4 | 1.000 | 1.000 | 0.011 / 0.010 |
| 16 | 0.982 | 0.909 | 0.012 / 0.010 |
| 64 | 0.673 | 0.324 | 0.010 / 0.010 |
| 256 | 0.126 | 0.022 | 0.012 / 0.011 |
| 1024 | 0.034 | 0.012 | 0.010 / 0.011 |
| unlimited | 0.010 | 0.010 | 0.010 / 0.010 |

Two things happen at once, and separating them is the point of the curve.

**The seen column is a memorisation-capacity curve.** The attacker holds ~4-16
distinct keystreams essentially perfectly, degrades through 64, and is exhausted
by 256-1024. This is a finite storage budget being consumed, not learning.

**The unseen column never moves.** It is 0.007-0.012 — chance — at *every*
level of key diversity, from 1 key to unlimited. Training on more keys produces
**no** transfer whatsoever to an unseen key.

This is a definitive negative answer to RQ2 in the zero-shot setting, and the
curve is what makes it defensible: the flat unseen line cannot be dismissed as
undertrained or under-capacity, because the same models demonstrably learn 16
keys at 0.98 accuracy. They can memorise; they cannot generalise across key
space. That is what the bound in `THESIS_ANALYSIS.md` §2 predicts — the
ciphertext does not identify which key produced it, and the chaotic
key->keystream map is non-smooth, so nothing interpolates.

The shapes dataset reproduces the pattern (linearmixer 1.000 -> 0.036 seen
across the same sweep; unseen flat at 0.008-0.013).

---

## 6. CBC feedback — corrected

An earlier reading of the partial data held that CBC feedback defeats this
attacker class *entirely*, in every setting. **The completed matrix refutes
that**, and the correction matters:

| config | dataset | attacker | ceiling |
|---|---|---|---|
| `diffuse_only_r1_cbc` | CIFAR-10 | **resunet** | **0.981** |
| `diffuse_only_r1_cbc` | CIFAR-10 | linearmixer | 0.037 |
| `diffuse_only_r1_cbc` | shapes | resunet | 0.020 |
| `diffuse_only_r1_cbc` | shapes | linearmixer | 0.023 |
| `feedback_cbc` (perm + CBC diffusion) | both | both | 0.009-0.010 |

A ResUNet on CIFAR-10 reaches a **0.981 same-key ceiling** on CBC diffusion. So
CBC does not defeat the attacker unconditionally — it defeats it on the
synthetic set and for the dense-linear attacker, but a convolutional attacker on
natural images inverts it almost perfectly. Richer plaintext statistics let the
network route around the exact modular arithmetic it cannot learn.

That cell then gives a *valid* resistance measurement: ceiling 0.981, unseen-key
**0.010**. CBC diffusion resists key-agnostic attack for a properly capable
attacker.

Where the ceiling *is* at chance (`feedback_cbc` on both datasets, CBC diffusion
on shapes), no security claim is made — those cells are reported as "no
information", not as resistance.

The mechanism evidence from F11 stands: the CBC inverse
`p_i = ((c_i XOR k_i) - c_{i-1}) mod 256` depends on two bytes, giving 65 536
combinations per position against 50 000 images, so the lookup is under-sampled
and the attacker must learn modular arithmetic. The memorisation control
confirmed both failure modes (training loss 0.25 -> 0.011 with 200 plaintexts,
retrieval still at chance on new plaintexts). What the completed sweep adds is
that **sufficient plaintext redundancy provides an alternative route**, which
the earlier shapes-only evidence could not show.

---

## 7. What the data does and does not support

| claim | supported? |
|---|---|
| Instrument detects a key-independent leak | yes — 1.000, +15.6 to +20.4 dB, both datasets |
| Instrument does not hallucinate | yes — AES and all mismatch controls at chance |
| **Row/column permutation leaks key-agnostically** | **yes — up to 22x chance, 2 datasets x 2 attackers** |
| Full 2-D permutation leaks much less | yes — 2.5-4.6x chance on CIFAR, ~1.5x on shapes |
| **Keyed diffusion supplies all key-agnostic security** | **yes — 1.000 -> 0.010 with permutation held key-independent** |
| Round count governs resistance | **no** — 1 round already at chance, 2-4 identical |
| float32 keystream degradation is exploitable | **no** — at chance |
| **Zero-shot key generalisation at any key diversity** | **no — flat at chance from 1 key to unlimited** |
| CBC feedback confers key-agnostic resistance | yes, where a ceiling exists (CIFAR/resunet 0.981 -> 0.010) |
| CBC feedback is "secure" | **not shown** — a claim about one attacker family, not the cipher |
| Attack success depends on attacker inductive bias | yes — 0.221 vs 0.087 on the same cipher |

**Scope.** Everything above concerns *zero-shot* key-agnostic attack at 32x32
with two attacker families and 5 000 training steps per run. It does not
establish that these configurations are secure against a key-adaptive
(known-plaintext) attacker, which is what `adapt.py` measures and which the §5
result specifically motivates.
