# Methodology findings

Confounds found while building the instrument. Each one, left in place, produces
an attack that fails for reasons unrelated to the cipher — and therefore a false
"this scheme resists neural cryptanalysis" result. They are recorded here
because *ruling them out is part of the contribution*: a negative cryptanalysis
result is only meaningful if the attacker was capable in the first place.

All numbers below are measured on this machine, not estimated.

---

## F1. A convolutional attacker cannot represent a position-dependent keystream

**Symptom.** Same-key attack (`n_keys=1`, pure-XOR diffusion, fixed key) stuck at
chance: top-1 retrieval @pool100 = 0.000 against a chance level of 0.010.

**Cause.** Convolution is translation-equivariant. A chaos keystream `k_i` is a
*pseudorandom function of absolute position*. A translation-equivariant network
applies the same filters everywhere and so cannot, even in principle, apply a
different XOR mask at each pixel.

**Fix.** Two normalised coordinate channels `(x, y)` appended to the input.

**Effect.** Same-key top-1 @pool100: `0.000 -> 0.904` (measured; and 0.904 was
itself the metric ceiling at the time — see F2 — so the true value was ~1.0).

**Consequence if ignored.** Every cipher would appear unbreakable, including a
cipher with no key at all.

**Scope.** This limit is real for a *position-dependent keystream*, and the
coordinate channels demonstrably fix it. It does not extend to permutation
inversion — see F10, where a U-Net inverts a global pixel gather completely.

---

## F2. Retrieval metric included the query in its own distractor pool

**Symptom.** Retrieval accuracy pinned at exactly 0.904 across different
architectures, representations, and training lengths — flat, and identical
between runs that should have differed.

**Cause.** Distractors were sampled uniformly from all `n` evaluation items,
including the query's own target. When the true target was drawn as its own
distractor the strict comparison `sim_true > sim_distractor` failed. The ceiling
is therefore

```
1 - (1 - 1/n)^(K-1)  =  1 - (999/1000)^99  =  0.906
```

which is what was being measured.

**Fix.** Mask self-matches out of the distractor set.

**Verification** (`retrieval_accuracy`, n=1000):

| predictor | top1@10 | top1@100 | top1@1000 |
|---|---|---|---|
| perfect (pred = target) | 1.0000 | 1.0000 | 1.0000 |
| constant (prior-only) | 0.1018 | 0.0109 | 0.0018 |
| random | 0.1014 | 0.0088 | 0.0001 |
| *chance* | *0.1* | *0.01* | *0.001* |

A perfect predictor now scores 1.0 and a prior-only predictor scores exactly
chance. Both endpoints are required for the metric to be trustworthy.

---

## F3. Input representation must match the cipher's algebra

**Symptom.** With bit-plane input, CBC-feedback diffusion stayed at chance even
in the same-key setting, while pure XOR under the same conditions succeeded.

**Cause.** The two operations live in different algebras:

- `XOR` is addition mod 2 — linear *only* in the bit-plane view.
- the CBC feedback term `(p_i + c_{i-1}) mod 256` is near-linear *only* in the
  pixel-value view; in bit-planes it is an 8-deep borrow chain.

An attacker given one representation cannot cheaply express the other.

**Fix.** `input_repr="both"` — pixel values and bit-planes concatenated (27
channels), plus coordinates.

**Consequence if ignored.** Any XOR/modular-arithmetic cipher looks strong
against a pixel-space attacker, and any modular-arithmetic cipher looks strong
against a bit-space attacker. Neither conclusion is about the cipher.

---

## F4. Smooth coordinates are a poor basis for a pseudorandom keystream

Coordinate channels (F1) are smooth, but `k_i` is high-frequency in position.
Added a free learnable per-position embedding (32 channels) so the attacker has
the *capacity* to represent an arbitrary position-dependent constant. It confers
no key knowledge: under many keys there is no fixed keystream to memorise.

---

## F5. Producer threads each built their own key pool

**Symptom.** Latent — no visible error.

**Cause.** Each of the 6 background encryption threads constructed its key pool
from a worker-offset seed, so `n_keys=k` silently produced `k x 6` distinct
keys, and only worker 0's pool coincided with the "seen" evaluation pool.

**Fix.** Separate `pool_seed` (shared: fixes *which* keys exist) from `seed`
(per-worker: fixes *which* draw). Verified: all 6 workers share one pool,
`n_keys=1` yields exactly 1 distinct key, seen/unseen pools disjoint.

**Consequence if ignored.** The key-diversity axis — the thesis's main
independent variable — would have been silently wrong in every run.

---

## F6. Shannon entropy of an encrypted 32x32 image cannot reach 8.0

Published chaos-encryption papers quote a target of ~7.99 bits. That target is
asymptotic. A 32x32x3 image supplies only 3072 samples across 256 bins, so the
finite-sample ceiling for a *perfectly uniform* source is lower.

Measured ceiling from ideal uniform data:

| image size | samples | entropy ceiling |
|---|---|---|
| 32x32x3 (CIFAR) | 3 072 | **7.9407** |
| 96x96x3 (STL-10) | 27 648 | 7.9932 |
| asymptotic | — | 8.0000 |

The implemented cipher measures **7.939** at 32x32, i.e. at the ceiling. Quoting
"7.94 < 7.99, therefore weak" would be an artifact of image size, not a
property of the cipher.

---

## F7. Every configuration needs its own same-key ceiling

A key-agnostic attack that fails is uninterpretable on its own: "the cipher
resists key-independent attack" and "this attacker cannot invert this cipher
under *any* circumstances" produce identical numbers.

Every cipher configuration is therefore run twice — at `n_keys=1` (same-key
ceiling) and `n_keys=None` (key-agnostic) — and the key-agnostic result is
reported *relative to its own ceiling*. A configuration whose same-key ceiling
is already at chance yields no security information at all and must be reported
as such rather than as a resistance result.

---

## F8. Plaintext redundancy is a separate resource from key recovery

Measured adjacent-pixel correlation:

| plaintext source | adjacent \|diff\| | adjacent correlation |
|---|---|---|
| uniform noise | 85.30 | 0.0006 |
| structured (`shapes`) | 5.97 | 0.9162 |

With uniform-random plaintexts there is no redundancy to exploit, so exact
inversion is the only available route. Running the identical attack on `noise`
and on structured plaintexts separates "the network inverted the cipher" from
"the network exploited plaintext structure". This is the sharpest available
answer to RQ1 and costs almost nothing to run.

---

## F9. An auxiliary SSIM loss term drives models *below* the prior floor

**Symptom.** On configurations the attacker could not solve, runs did not settle
at the prior floor as an unsolvable task should. They diverged *past* it: loss
rising 0.478 -> 0.566 while PSNR fell 7.36 -> 6.03 dB against a floor of 11.09.

**Cause.** `L = L1 + 0.15*(1 - SSIM)`. A constant prediction has zero local
variance, so its SSIM is degenerate and the SSIM term is maximally penalised.
The optimiser manufactures spurious structure to escape that penalty, which
costs L1. For reference on this dataset:

| predictor | PSNR |
|---|---|
| per-pixel mean (prior floor) | 11.13 |
| constant 0.5 | 11.13 |
| constant 0.0 | 5.21 |
| the diverged model | ~6.0 |

i.e. the model had been pushed almost to a saturated-black output.

**Fix.** Plain L1 for training. SSIM is still reported as an evaluation metric.

**Consequence if ignored.** Sub-floor PSNR is not a stronger negative result --
it is a broken optimisation. It also makes `gain over floor` negative for
reasons unrelated to the cipher, corrupting the headline comparison.

---

## F10. Correction to F1: convolution *can* invert a global pixel gather

An earlier reading of the F1 evidence held that a convolutional attacker could
not, even in principle, invert a full 2-D pixel permutation, and that the
`LinearMixer` was therefore required for permutation-dominant ciphers.

That was too strong, and the measurements refute it. Once the loss was changed
to L1 (F9), ResUNet solves the key-independent permutation control completely:

| attacker | regime | seen top1@100 | unseen top1@100 | gain over floor |
|---|---|---|---|---|
| ResUNet | same-key | 1.000 | 1.000 | +14.22 dB |
| ResUNet | key-agnostic | 1.000 | 1.000 | +13.86 dB |
| LinearMixer | same-key | 1.000 | 1.000 | +15.49 dB |
| LinearMixer | key-agnostic | 1.000 | 1.000 | +15.50 dB |

The real obstacle was the optimisation pathology of F9, not the inductive bias.
Coordinate channels plus a learnable positional embedding give the U-Net enough
positional addressing to express the gather through its bottleneck.

LinearMixer is still kept and still reaches a higher gain (+15.5 vs +13.9 dB),
and running both remains the right protocol -- a resistance claim should be made
against the stronger of several attacker classes, not against one architecture.
But the inductive-bias argument must be stated as a *quantitative* difference,
not an impossibility claim.

---

## F11. CBC modular feedback defeats this attacker class entirely — and why

**Result.** With every other variable held fixed (dataset, single fixed key,
architecture, input channels, loss, steps), the diffusion feedback mode alone
decides whether the attack works at all:

| feedback | seen top-1 @100 | seen PSNR | vs floor |
|---|---|---|---|
| none (XOR keystream) | **1.000** | 25.37 | +14.2 dB |
| CBC chain | **0.011** (chance) | 11.10 | −0.08 dB |

Three independent attempts to lift the CBC ceiling all failed:

1. **Flat-raster predecessor channel** — supplying `c_{i-1}` directly, correctly
   aligned across the channel boundary. Result: chance (0.011).
2. **256-way categorical head** — cross-entropy over byte values, so the jump
   discontinuity in the inverse is representable. Head verified exact (0.0 loss
   on perfect logits, ln 256 = 5.545 on uniform). Result: loss pinned at
   **exactly 5.543–5.551**, i.e. a uniform posterior. Zero information extracted.
3. **Longer training / both attacker architectures.** Result: chance.

**Mechanism.** The inverse is `p_i = ((c_i XOR k_i) - c_{i-1}) mod 256`, a
function of *two* ciphertext bytes rather than one. Per pixel position that is
65 536 input combinations, against 50 000 training images — the lookup is
under-sampled, so the attacker cannot memorise it and must instead learn modular
arithmetic. Pure XOR is `p_i = c_i XOR k_i`, only 256 combinations per position,
which is comfortably memorised as a per-position table.

**Confirmed by the memorisation control.** Repeating the CBC run with the
training set cut to 200 distinct plaintexts makes the lookup memorisable, and
the training loss duly collapses (0.25 → 0.025 — the model now fits what it is
shown). But retrieval on *new* plaintexts under the *same* key stays at chance
(0.008–0.014) with PSNR *below* the prior floor. So:

| training plaintexts | fits training data | generalises to new plaintexts |
|---|---|---|
| 50 000 | no (loss ~0.25) | no |
| 200 | yes (loss ~0.025) | no |

Both failure modes are exhibited. The network memorises when it can and learns
nothing transferable when it cannot: **it never acquires the modular rule.**

**Consequence for the study design.** A cipher whose same-key ceiling is at
chance yields no security information (F7). Using CBC as the *baseline* would
therefore silently void every other axis — rounds, permutation scope, keystream
precision. The baseline was changed to `feedback="none"`, which has a verified
live ceiling (CIFAR-10: seen 1.000, +12.1 dB over floor), and CBC is retained as
an explicit variant so this effect is reported rather than hidden.

**How this must be stated.** "CBC feedback resists regression- and
classification-based neural cryptanalysis, including in the same-key setting"
is supported. "CBC feedback is secure" is **not** — this measures one attacker
family, and the limitation is plausibly the well-documented difficulty neural
networks have learning modular arithmetic, not a property of the cipher.

---

## F12. Key-specific memorisation does not survive even a handful of keys

The key-diversity sweep (CIFAR-10, baseline cipher) shows the collapse is
immediate rather than gradual:

| distinct training keys | seen top-1 (keys **in** the training pool) | unseen top-1 |
|---|---|---|
| 1 | **1.000** | 0.008 |
| 16 | 0.010 | 0.010 |
| 64 | 0.010 | 0.010 |
| 256 | 0.010 | 0.010 |

With one key the attacker inverts the cipher completely. With sixteen it fails
on keys it *trained on*.

This is the §2 bound made concrete. The ciphertext does not identify which key
produced it, so an attacker with no key input cannot select the right keystream
to subtract; and the chaotic key→keystream map is non-smooth, so nothing
interpolates across key space. What looked like "learning to invert the cipher"
at `n_keys=1` is entirely key-specific memorisation, and it does not survive
contact with a second key.

This is a strong negative answer to RQ2 in the zero-shot setting, and it is
precisely the reason the few-shot key-adaptive setting (`adapt.py`) exists: it
supplies the key information the zero-shot attacker provably lacks.
