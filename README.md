# Crynexa

**Key-agnostic neural cryptanalysis of chaos-based image encryption.**

Can a neural network learn the *encryption transformation itself*, rather than
memorising one secret key? This repository contains a reproducible framework
that measures the answer, and the results of **223 trained attack models**.

Every number in this repository was measured from a trained model. None are
estimated or simulated.

---

## The question

Chaos-based image ciphers are usually evaluated with statistical tests (entropy,
NPCR, UACI). Those describe the *ciphertext*. They say nothing about whether a
data-driven attacker can invert the scheme.

Treat the cipher as a black box `C = E_K(P)` and train an attacker
`P̂ = F_θ(C)` across many secret keys, then test it on keys it has **never
seen**. If reconstruction survives, the attacker learned something
*key-independent* about the cipher's structure.

There is a hard limit worth stating up front: for a cipher with a large key
space and a high-entropy keystream, marginalising out an unobserved key makes
the ciphertext statistically independent of the plaintext, so the optimal
attacker outputs the dataset mean. Zero-shot key-agnostic recovery is therefore
**bounded, not merely hard** — and how far a real scheme sits from that bound is
exactly what this framework measures.

---

## In plain English

**The problem.** People scramble images so nobody else can see them. One popular
family of methods uses *chaos* — maths that turns a secret password into a stream
of unpredictable numbers, then uses those numbers to shuffle the pixels and change
their colours. Normally you need the password to unscramble it.

Our question: **if an AI studies thousands of scrambled-and-original image pairs,
can it learn to unscramble a picture locked with a password it has never seen?**
If so, an attacker would not need to steal your password at all.

**What we built.** We wrote the encryption ourselves with a switch for each
security feature, so we could turn them on and off one at a time and see which one
actually does the protecting. Then we trained **223 AI attackers** against it.

The important part is how we check our own work. Every test also asks: *can this AI
unscramble the image when we DO give it the password?* If it cannot even do that,
then failing without the password proves nothing — the AI was simply too weak. We
report those cases as **"no information"** rather than claiming the encryption is
safe. We also included bank-grade AES, which the attack **must** fail against, and
a deliberately passwordless cipher, which it **must** break. Both behaved
correctly, so the measuring instrument is trustworthy.

**What we found.**

1. **A common shortcut is genuinely breakable.** Some schemes shuffle rows and
   columns separately because it is faster. That leaves a fingerprint identical for
   every password — so our AI recovered pictures locked with passwords it had never
   seen.
2. **Shuffling barely protects anything; the colour-changing step protects
   everything.** With shuffling alone the attack succeeded completely. Adding one
   password-controlled colour-changing step dropped it to pure guesswork.
3. **The AI never truly "learns the cipher".** It memorises about 16 passwords
   almost perfectly, then transfers *nothing* to password 17. More training
   passwords did not help at all.
4. **More rounds do not help.** Repeating the encryption 2, 3 or 4 times was no
   safer than once.

**The clearest single picture.** We locked 1000 test photos with a brand-new
password and measured where the correct photo ranked among 1000 possibilities.
Rank 1 = picked it outright. Rank ~500 = guessing.

| encryption used | where the right photo ranked | meaning |
|---|---|---|
| fake cipher, no password protection | **1st** of 1000 | completely broken, as expected |
| **rows & columns shuffled separately** | **53rd** of 1000 | **really is leaking — top 5%** |
| proper shuffling + password colour change | 502nd of 1000 | guesswork — held up |
| AES (bank-grade) | 501st of 1000 | guesswork, as expected |

**Practical takeaway.** Do not shuffle rows and columns separately to save time,
and never rely on shuffling alone — the password-driven colour-changing stage is
what actually keeps the image secret.

**Honest limit.** We tested an attacker with *no* knowledge of the password. We did
not test one that already holds a few matching original/scrambled examples for that
exact password — an easier situation, and nothing here says these schemes are safe
against it.

## Headline results

| finding | evidence |
|---|---|
| **Row/column permutation is broken without the key** | unseen-key top-1 retrieval **0.221** vs 0.010 chance (22×), replicated on 2 datasets × 2 attacker families, all mismatch controls at chance |
| **Keyed diffusion supplies all key-agnostic security** | with the permutation held *fully key-independent*: no diffusion → **1.000**, one keyed diffusion stage → **0.010** |
| **Zero key generalisation at any key diversity** | unseen-key retrieval pinned at chance from 1 training key to unlimited, while the same models memorise ~16 keys at 0.98 |
| **Round count and keystream precision do not matter** | rounds 1–4 and float32 vs float64 all at chance, each with a valid same-key ceiling |

A single end-to-end demonstration on one freshly drawn key (1000 held-out
CIFAR-10 images, median rank of the true plaintext among 1000 candidates):

| cipher | median rank | top-1 | PSNR gain |
|---|---|---|---|
| key-independent permutation *(positive control)* | **1** / 1000 | 1.000 | +20.5 dB |
| **row/column permutation** | **53** / 1000 | 0.062 | +2.5 dB |
| keyed permutation + keyed diffusion | 502 / 1000 | 0.001 | −0.0 dB |
| AES-128-CTR *(negative control)* | 501 / 1000 | 0.001 | −0.0 dB |

Full analysis: [`docs/RESULTS_FINAL.md`](docs/RESULTS_FINAL.md).

## Why the methodology is the contribution

A cryptanalysis result that *fails* only means something if the attacker was
capable in the first place. Two failures look identical in the numbers:

- the cipher resists key-independent attack, and
- this attacker cannot invert the cipher **at all**.

So every cipher configuration is run twice — once with a single fixed key (the
**same-key ceiling**) and once with unlimited key diversity — and the
key-agnostic number is only interpreted relative to its own ceiling. Cells whose
ceiling is at chance are reported as *no information*, never as resistance.

Five controls back every number: a prior floor (constant = mean training image),
a mismatch control (wrong ciphertext, to quantify hallucination from the image
prior), an **AES-CTR negative control** that must fail, a **key-independent
positive control** that must break, and the per-configuration ceiling.

Retrieval accuracy — not PSNR — is the primary metric. PSNR rewards outputs that
are *plausible*; only retrieval distinguishes "recovered **this** image" from
"produced **a** plausible image".

[`docs/METHODOLOGY_FINDINGS.md`](docs/METHODOLOGY_FINDINGS.md) documents twelve
confounds found during development, each of which would have manufactured a
false security result — including two corrections to earlier conclusions.

## Quick start

```bash
pip install torch torchvision numpy numba pillow pyarrow cryptography matplotlib

python tests/test_cipher.py          # 576 cipher configs + all controls
python run_experiment.py --name demo --device cuda:0 \
  --json '{"dataset":"cifar10","n_keys":1,"steps":2000}'

python sweep.py --suite E1 --gpus 0,1 --dataset cifar10 --tag _cifar --skip-done
python sweep.py --suite E2 --gpus 0,1 --dataset cifar10 --tag _cifar --skip-done
```

### Results frontend

```bash
python build_web.py && python demo_run.py   # rebuild payloads
./serve.sh                                  # http://localhost:8010/
```

A React app (vendored, no build step) covering methodology, datasets, cipher
validation, calibration, the key-diversity curve, an end-to-end sample run, and
every individual result.

## Deploying the frontend

The site is plain static files with React vendored in, so there is no build step.
`vercel.json` points Vercel at `web/` — import the repository on Vercel and it
deploys as-is, no configuration needed.

Locally:

```bash
python build_web.py && python demo_run.py   # regenerate the data payloads
./serve.sh                                  # http://localhost:8010/
```

## Layout

```
crynexa/chaos/      parameterised chaos cipher (4 maps, 8 knobs) + AES-CTR control
crynexa/data/       on-the-fly plaintext-ciphertext generation, uniform key sampling
crynexa/models/     ResUNet, LinearMixer, PixelMixer, PlainCNN attackers
crynexa/metrics/    retrieval, PSNR/SSIM, controls, KGG, cipher statistics
crynexa/train.py    single-run training and evaluation
sweep.py            experiment matrix across GPUs (resumable via --skip-done)
adapt.py            few-shot key-adaptive (known-plaintext) attack
demo_run.py         end-to-end demonstration under a freshly drawn key
web/                results frontend
docs/               analysis, methodology findings, final results
runs/*/result.json  the measured record of all 223 runs
```

Model weights (19 GB) and datasets are not committed; every run is reproducible
from the config stored in its `result.json`.

## Cipher validation

The implemented scheme meets published chaos-cipher targets, so attack results
are not measured against a strawman:

| statistic | measured | target |
|---|---|---|
| Shannon entropy | 7.940 | 7.9407 *(finite-sample ceiling at 32×32, not 8.0)* |
| adjacent correlation | +0.0061 | ≈ 0 |
| NPCR | 99.515 % | ≈ 99.61 % |
| UACI | 33.394 % | ≈ 33.46 % |

All 576 cipher configurations round-trip exactly (`tests/test_cipher.py`).

## Scope

These results concern *zero-shot* key-agnostic attack at 32×32 with two attacker
families. They do **not** establish security against a key-adaptive
(known-plaintext) attacker — that is a separate setting, implemented in
`adapt.py`, which supplies the key information the zero-shot attacker provably
lacks.
