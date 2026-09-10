# Key-Agnostic Neural Cryptanalysis of Chaos-Based Image Encryption
## Deep analysis, feasibility assessment, and implementation plan

---

## 0. Verdict up front

**Feasible: yes.** As a master's thesis this is a strong, well-scoped, defensible project. Compute is not a constraint (4× L40S 46 GB available; the entire experimental program is a few GPU-days). Data is self-generated, so there is no data-acquisition risk. The literature gap is real.

**But the proposal as written contains one methodological flaw that is close to fatal, and three design decisions that will cost months if left uncorrected.** All four are fixable now, cheaply, before a line of code is written. They are covered in §2 and §3.

The short version of the fix:

> The zero-shot key-agnostic attack described in the proposal is, for a correctly-implemented cipher, **information-theoretically impossible** — not hard, impossible. The thesis must therefore be restructured so that this impossibility is a *measured result* rather than a *failed goal*, and a second, well-posed attack setting (few-shot key adaptation) must be added so the thesis also produces positive results.

That restructure turns a high-risk thesis into a low-risk one with guaranteed publishable output regardless of which way the experiments go.

---

## 1. What the proposal gets right

These decisions are sound and should be kept without change:

| Element | Assessment |
|---|---|
| Framing as *cryptanalysis*, not *decryption* (§8 of the proposal) | Correct and important. Keep this discipline throughout. |
| Self-generated plaintext–ciphertext dataset | Correct. Removes all dataset-availability risk and gives full control over the independent variables. |
| Implementing the cipher yourself rather than using a third-party implementation | Correct. You need parameter-level control to run ablations; a black-box implementation makes the interesting experiments impossible. |
| Three-tier dataset ladder (CIFAR-10 → STL-10 → DIV2K) | Correct, and the cross-dataset test genuinely separates "learned the cipher" from "learned image statistics". |
| U-Net family for an image-to-image inverse | Correct default for the diffusion-dominant case. (Not for the permutation-dominant case — see §3.4.) |
| Introducing a Key Generalization Gap metric | Good instinct. Needs a floor term to be meaningful — see §4.3. |
| Attack-success-vs-data curve (§20 of the proposal) | Good instinct, but measuring the wrong x-axis. See §4.4. |
| Three research questions | Well-formed. RQ2 needs one clause added. See §6. |

---

## 2. The central problem: the zero-shot cross-key attack is ill-posed

### 2.1 Statement

The proposal's primary experiment (Experiment B) is:

- Train on ciphertexts under keys `K_train`
- Test on ciphertexts under keys `K_test`, with `K_train ∩ K_test = ∅`
- The attacker network receives **only the ciphertext** `C*` and must output `P̂ ≈ P*`

Consider what the optimal such network computes. Trained under a pixel-wise reconstruction loss, the Bayes-optimal predictor is the posterior mean:

```
F*(C*) = E[ P | C = C* ]
```

where the expectation is over the joint posterior induced by the plaintext prior **and the unknown key**, since the key is not observed at test time.

Now suppose the cipher is well-constructed: the keystream is a high-entropy function of the key, and the key space is large. Then `C*` is statistically independent of `P*` when `K` is unknown and uniformly distributed. The posterior collapses to the prior:

```
E[ P | C = C* ]  =  E[ P ]  =  the mean image of the dataset
```

The optimal attacker is a **constant function** that outputs a gray blob. Not because the network is too small, or the training set too small, or the architecture wrong — because there is no information in `C*` about `P*` once `K` is marginalised out. No amount of training keys, data, or model capacity changes this.

### 2.2 Why this is not a reason to abandon the thesis

Because real chaos-based image encryption schemes are **not** well-constructed in that sense, and the *degree* to which they fail is exactly the quantity worth measuring.

`C*` retains information about `P*` under an unknown key to the extent that the cipher contains **key-independent structure**. The main leakage channels, ranked by how exploitable they are:

1. **Permutation-only or permutation-dominant schemes.** If `C` is a pixel permutation of `P`, the multiset of pixel values is preserved *exactly*, for every key. The histogram is key-independent. Reconstruction reduces to jigsaw assembly under a natural-image prior — genuinely solvable key-agnostically, and a known weakness class in the literature.

2. **Row-wise / column-wise permutation.** Many published chaos schemes permute rows and columns separately rather than performing a full 2-D scramble. This preserves the multiset of entire rows and columns. Leakage is enormous and key-independent.

3. **Insufficient rounds.** A single permutation–diffusion round leaves local correlation structure that survives key marginalisation. This is the cleanest knob to sweep.

4. **Finite-precision keystream degradation.** Chaotic maps implemented in float64 have shorter effective cycles and non-uniform keystream statistics than the idealised map. If the keystream distribution is biased in a key-independent way, a network can learn and partially subtract that bias.

5. **Block-local operation.** Schemes operating on 8×8 or 16×16 blocks confine mixing within blocks, preserving inter-block relationships regardless of key.

Conversely, one property of chaotic maps works **against** the attacker and is worth stating explicitly in the thesis: chaotic maps are designed for extreme initial-condition sensitivity. Two nearby keys produce completely decorrelated keystreams. Therefore the map `K → keystream` is **maximally non-smooth**, and a network *cannot interpolate across key space*. This is the theoretical reason why "train on 70 keys, generalise to key 71" is hopeless — and it is a genuinely interesting thing to state and demonstrate.

### 2.3 The reframe

Do not pose the thesis as *"can a network break chaos encryption without the key?"* — that is a yes/no gamble on an experiment whose answer is probably "no".

Pose it as:

> **Key-agnostic recoverability is a measurable property of a cipher configuration. This thesis builds an instrument that measures it, and uses that instrument to identify which structural properties of chaos-based image encryption make it vulnerable to key-independent neural attack.**

Under this framing:

- If the attack fails on a strong configuration → **that is the result**: quantified evidence of key-mixing soundness.
- If the attack succeeds on a weak configuration → **that is the result**: an identified structural weakness with a dose-response curve.
- Either way you have a thesis, and both outcomes are guaranteed to occur across a configuration sweep.

This is the single most important change to make.

---

## 3. Four design corrections

### 3.1 The key partition scheme (K001–K100) is wrong

The proposal partitions 100 enumerated keys into train/val/test. Two problems:

**Problem A — 70 keys is nowhere near enough to hope for key-space generalisation.** The nominal key space of a logistic-map scheme with key `(x₀, r)` in float64 is on the order of 2⁵⁰. Seventy samples from it carry no information about the seventy-first.

**Problem B — enumerated keys are almost certainly not uniformly sampled.** If `K001…K100` are generated as `x₀ = 0.01, 0.02, …` or from a seeded sequential counter, they occupy a structured, measure-zero subspace of the key space. Any "generalisation" observed would be an artifact of that structure and would not survive review.

**Fix.** Encryption is cheap — a few hundred microseconds per 32×32 image in NumPy. Do not pre-generate a fixed key set at all. **Sample a fresh key uniformly from the full key space for every training example, on the fly, inside the dataset's `__getitem__`.** This gives effectively unlimited key diversity for free, and makes "unseen key" automatic: the probability that a test key was seen in training is ~0.

The finite-key experiment then becomes a **deliberate ablation** rather than the default setup:

```
N_keys ∈ {1, 4, 16, 64, 256, 1024, ∞}
```

Sweeping `N_keys` and plotting held-out-key performance produces the single most important figure in the thesis (see §4.4, Curve 1). At `N_keys = 1` you reproduce the same-key baseline; at `N_keys = ∞` you measure pure key-agnostic recoverability. It is the same experiment as the proposal's Experiment A and Experiment B, unified into one continuous axis. That is a much stronger presentation than two disconnected experiments.

**Corollary:** the plaintext train/test split must also be disjoint, or the network memorises plaintexts and the retrieval metric becomes meaningless.

### 3.2 The proposal has no control experiments — this is the flaw reviewers will attack

Given a 32×32 CIFAR ciphertext and a U-Net trained on CIFAR, the network can produce a plausible-looking small colour image *purely from the dataset prior*, with no information extracted from the ciphertext whatsoever. It will score a respectable PSNR. It will look convincing in a figure. It will be worthless.

Any examiner familiar with generative models will raise this in the first ten minutes of the defence. Four controls are required, and all are cheap:

**Control 1 — Prior floor.** Compute the metrics for the constant predictor `P̂ = mean training image`. This is the PSNR/SSIM value corresponding to *zero information extracted*. Every reported attack number must be presented against this floor. Report the gain over floor, not the raw value.

**Control 2 — Mismatched-ciphertext control.** Feed the trained attacker a ciphertext belonging to a *different* plaintext and score the output against the original plaintext. Any score above the prior floor here is pure hallucination from the image prior. This directly quantifies the hallucination component.

**Control 3 — AES-CTR negative control.** Implement AES-CTR as one of the selectable ciphers and run the entire attack pipeline against it. The attack **must** collapse to the prior floor. If your pipeline appears to "break" AES, you have a bug — a leaked plaintext channel, a key reused between train and test, or a normalisation that carries plaintext information. This is a five-line addition and it is the cheapest insurance in the whole project. Include the AES row in every results table; it is what makes the other rows believable.

**Control 4 — Noise-plaintext control.** Run the attack with uniformly random plaintexts instead of natural images. There is no image prior to exploit, so *any* reconstruction above chance is unambiguously cryptanalytic information rather than learned image statistics. This is the cleanest possible evidence that the attack is real, and it is nearly free to run.

### 3.3 PSNR and SSIM are the wrong primary metric — use retrieval

PSNR and SSIM reward outputs that are *plausible*. They do not distinguish "recovered **this** image" from "produced **a** natural-looking image". For a cryptanalysis claim, that distinction is the entire point.

**Primary metric: top-1 retrieval accuracy.** Given a reconstruction `P̂ᵢ` and a candidate pool of `N` plaintexts (the true `Pᵢ` plus `N−1` distractors from the held-out set), rank candidates by similarity to `P̂ᵢ` and record whether the true plaintext ranks first. Report at `N ∈ {10, 100, 1000}`. Chance is `1/N`.

This metric is unforgeable by a prior-only model: a gray blob is equidistant from every candidate and scores exactly at chance. A model that has genuinely extracted information about *which* image was encrypted scores above chance even when its PSNR is poor. It is the correct instrument for the question being asked.

Keep PSNR/SSIM/MSE as secondary reconstruction-quality metrics. Report both. Lead with retrieval.

Also note: **NPCR and UACI are properties of the cipher, not of the attack.** They measure plaintext sensitivity of the encryption. They belong in the cipher-characterisation section (establishing that your implemented cipher meets published standards and is not a strawman) and must not be presented as attack-success metrics. The proposal's §18 currently conflates these; separate them cleanly.

### 3.4 A convolutional U-Net is the wrong architecture for the permutation-dominant case

Convolution assumes spatial locality is meaningful. A full 2-D pixel permutation destroys locality by construction: after scrambling, adjacent ciphertext pixels are unrelated, and the pixels that belong together are arbitrarily far apart. A CNN's inductive bias is actively counterproductive here — this is why permutation-only ciphers may appear "unbreakable" to a U-Net while being trivially broken by an appropriate method.

This is worth stating in the thesis as a finding in its own right: *attack success is a joint property of the cipher configuration and the attacker's inductive bias, not of the cipher alone.*

Practically: run the U-Net everywhere as the standard attacker, and add one permutation-appropriate model (a small ViT or set-transformer over pixel/patch tokens, which is permutation-tolerant) for the permutation-dominant configurations. Do not build a model zoo — two well-chosen architectures making an inductive-bias point is worth more than five architectures compared on a leaderboard.

---

## 4. The corrected experimental design

### 4.1 The cipher as a parameterised instrument

The single highest-value engineering decision in the project: **do not implement one fixed cipher.** Implement a cipher family with explicit, independently controllable knobs. The knobs are the independent variables of the entire thesis.

```
ChaosCipher(
    map        = logistic | tent | henon | pwlcm | arnold,
    rounds     = 1 .. 8,
    mode       = permute_only | diffuse_only | permute_diffuse,
    perm_scope = full2d | rowcol | blockwise(B),
    perm_keyed = True | False,          # key-dependent vs fixed permutation
    feedback   = none | cbc,            # plaintext-dependent diffusion
    precision  = float32 | float64,     # keystream degradation
    key        = (x0, r, ...) sampled uniformly from the key space
)
```

Plus `AESCTR(key)` exposing the identical interface, as the negative control.

With this, the thesis question becomes a dose–response study — *which knob settings produce key-agnostic recoverability, and how much* — instead of a single pass/fail experiment. This is what makes the work publishable rather than merely complete.

Anchor the parameter defaults to a specific published chaos-encryption scheme and cite it, so that at least one configuration in your sweep is a faithful reproduction of a real proposed cipher. This forecloses the "you attacked a strawman" objection.

### 4.2 Two attack settings, not one

**Setting 1 — Zero-shot key-agnostic (the proposal's original).**
Attacker sees only `C*`. This is the setting analysed in §2. Expected to fail for strong configurations and succeed for structurally weak ones. Measures *key-independent structural leakage*.

**Setting 2 — Few-shot key-adaptive (new; add this).**
At test time the attacker is given `n` known plaintext–ciphertext pairs generated under the *unseen* key `K*`, then must reconstruct a *new* ciphertext under `K*`.

This setting is **well-posed** — the support pairs supply the missing key information, so the posterior does not collapse — and it corresponds to the classical known-plaintext attack model, making it more cryptographically meaningful than the zero-shot setting anyway. It is also where the positive results will come from, which is why it is the project's insurance policy.

Three implementations, in increasing order of novelty and risk:

- **(a) Fine-tune baseline.** Take the zero-shot-trained network, fine-tune on the `n` support pairs, evaluate on a held-out ciphertext under `K*`. Trivial to implement, and it alone answers the practically interesting question: *how many known plaintexts does an attacker need to break a fresh key?* Build this first; it is the safety net.
- **(b) In-context conditioning.** Feed the support pairs to the network alongside the query ciphertext (extra channels, or cross-attention over a support encoder) so adaptation happens in a single forward pass with no gradient steps. This is the genuinely novel contribution — meta-learned / amortised cryptanalysis.
- **(c) Hypernetwork.** A support encoder emits FiLM parameters or weights for the reconstruction network. Higher risk, only if (b) works and time remains.

Do (a) unconditionally. Do (b) as the thesis's methodological contribution. Treat (c) as optional.

Pre-training on many keys pays off here even when zero-shot fails: a network that has seen millions of keys should adapt to a new key from far fewer support pairs than one trained from scratch. **That sample-efficiency gain is itself a clean, quantitative demonstration that the model learned something key-independent about the cipher family** — which is precisely RQ2, answered positively, via a route that does not depend on the impossible zero-shot result. This is the cleverest part of the restructured design and should be a headline claim.

### 4.3 Corrected Key Generalization Gap

The proposal's `KGG = Perf(seen) − Perf(unseen)` is uninterpretable without a floor: a 4.4 dB gap means something entirely different when the floor is 12 dB versus 25 dB.

Normalise against the prior floor:

```
KGG_norm = (S_seen − S_unseen) / (S_seen − S_floor)
```

where `S_floor` is Control 1 (§3.2). Then:

- `KGG_norm = 0` → attack transfers perfectly to unseen keys (complete key-agnostic break)
- `KGG_norm = 1` → attack retains nothing on unseen keys (key mixing sound; the network memorised keys)

A bounded, unit-free quantity, comparable across ciphers, datasets, resolutions and architectures. Report the raw gap too, but define the normalised form as the headline metric.

### 4.4 The four figures the thesis is built around

**Curve 1 — Key diversity → zero-shot unseen-key performance.**
x: number of distinct training keys (1 → ∞, log scale). y: retrieval accuracy and PSNR on held-out keys, with the prior floor drawn as a horizontal line.
*This is the money plot for RQ2.* Flat at the floor = strong negative result. Rising = structural leak identified.

**Curve 2 — Known pairs under the target key → adapted performance.**
x: `n` support pairs (0, 1, 10, 100, 1 000, 10 000). y: performance on new ciphertexts under that key. One line per pre-training regime (from scratch / pre-trained on many keys).
*The practical attack curve, and the sample-efficiency evidence for RQ2.*

**Curve 3 — Cipher strength → attack success.**
x: rounds, or mode, or permutation scope. y: attack success in both settings. AES-CTR as the bottom reference line.
*This is the actual scientific contribution: which structural property confers resistance.*

**Curve 4 — Transfer matrix.** A heatmap: trained-on configuration × tested-on configuration (logistic/tent/Hénon × CIFAR/STL-10/DIV2K). Diagonal = matched; off-diagonal = transfer. Covers the proposal's Experiments C and D in a single figure.

---

## 5. Feasibility assessment

### 5.1 Compute — not a constraint

Measured on this machine: 4× NVIDIA L40S (46 GB each), 503 GB RAM, 256 GB free disk, PyTorch 2.12 + CUDA available.

| Workload | Estimate |
|---|---|
| U-Net on CIFAR-10 32×32, one config to convergence | minutes on one GPU |
| Full `N_keys` sweep (7 points × 3 seeds) on CIFAR | a few GPU-hours |
| STL-10 96×96 | comfortable |
| DIV2K 256×256 patches | comfortable; the largest single job in the project |
| Entire experimental programme, all sweeps, all seeds | on the order of GPU-days, parallel across 4 GPUs |

This project is compute-light by modern standards, and the hardware available is roughly two orders of magnitude more than required. **Do not spend thesis time on training optimisation** — it is not a bottleneck and it is not where the marks are.

### 5.2 Storage — solved by not storing anything

The proposal's 500 000-sample pre-generated dataset is unnecessary and, given §3.1, actively harmful (it forces a finite key set). Generate ciphertexts on the fly in the dataloader. Persist only:

- the fixed evaluation sets, serialised with their keys, for exact reproducibility
- trained checkpoints
- metrics/figures

Total footprint: a few GB, against 256 GB free. Not a constraint.

### 5.3 Cipher throughput — the one real engineering risk

On-the-fly generation makes the CPU the bottleneck if the cipher is written as a naive Python pixel loop. A per-image Python loop over 3 072 pixels will starve four L40S GPUs.

Requirement: implement the chaotic sequence generation and the permutation/diffusion as **vectorised NumPy over a whole batch at once**, or as a small Numba/Torch kernel. Keystream generation is inherently sequential per sequence, but sequences for *different images* are independent and therefore trivially batchable along the batch axis. Budget two days for this and benchmark it before building anything else: target ≥ 5 000 images/second for 32×32.

Fallback if this proves awkward: pre-generate a large-but-finite key pool (10⁵–10⁶ keys) rather than truly unlimited. Statistically equivalent for all purposes here.

### 5.4 Scientific risk — mitigated by the reframe

| Risk | Severity if unmitigated | Mitigation |
|---|---|---|
| Zero-shot cross-key attack fails entirely | Would be fatal under the original framing | §2.3 reframe makes failure a result; §4.2 Setting 2 guarantees positive results |
| Examiner: "the network only learned the image prior" | Fatal to the central claim | The four controls in §3.2 and the retrieval metric in §3.3 |
| Examiner: "you attacked a strawman cipher" | Serious | Reproduce a named published scheme; report its NPCR/UACI/entropy against published values; include AES-CTR |
| Cited prior work already did this | Serious | See §7 — verify precisely what "key-independent" meant in those papers before committing |
| Scope creep into building the GUI framework | Very likely; costs 4–6 weeks | See §5.5 — cut it |

### 5.5 Cut the web/desktop framework

The proposal's §24 specifies "NeuroChaosBreak", a web or desktop application with dropdowns for algorithm, chaos map, key configuration, dataset, model and training configuration.

**Cut it.** It is the largest single time sink in the plan and it earns close to zero thesis marks. A GUI is a presentation layer over experiments that are configured far more reproducibly by a YAML file, and every hour spent on the front end is an hour not spent on the experiments that constitute the actual contribution.

Replace it with:

- a CLI driven by YAML configs (`python -m crynexa.run --config configs/expB_logistic.yaml`)
- an automatic report generator emitting the figures and LaTeX-ready tables straight into the thesis

That satisfies the "reproducible framework" contribution *better* than a GUI does, because a config file is a citable artifact and a dropdown is not. If a demo is genuinely required for the defence, a ~100-line Streamlit page wrapping the finished pipeline can be added in the final week — after the science is done, and only then.

---

## 6. Corrected research questions and hypotheses

**RQ1.** Can a deep neural network reconstruct plaintext images from chaos-based ciphertexts without recovering the secret key, and how much of any apparent reconstruction is attributable to the learned image prior rather than to cryptanalytic information?

*(The second clause is the addition. It commits you to the controls and pre-empts the strongest objection.)*

**RQ2.** To what extent does a model trained across a large, diverse key population acquire key-independent knowledge of the encryption transformation — measured both by zero-shot performance on unseen keys and by sample efficiency when adapting to an unseen key from a limited number of known plaintext–ciphertext pairs?

*(Adds the few-shot route, so RQ2 is answerable positively even when zero-shot fails.)*

**RQ3.** Which structural properties of a chaos-based image cipher — number of rounds, permutation scope, presence of keyed diffusion, plaintext feedback, keystream precision — govern its resistance to neural cryptanalysis, and how does attack effectiveness transfer across chaos maps and image distributions?

*(Reframed from "how does it change" to "which properties govern it": a mechanism question, not a description.)*

**H1.** Reconstruction quality above the prior floor is achievable for cipher configurations containing key-independent structure, and collapses to the floor for configurations with full keyed mixing and sufficient rounds.

**H2.** Pre-training across a diverse key population confers no meaningful zero-shot advantage on unseen keys — because chaotic key-to-keystream maps are non-smooth — but does confer a measurable sample-efficiency advantage when adapting to an unseen key, demonstrating that key-independent structure of the cipher family has been learned.

**H3.** Attack effectiveness is a joint function of cipher configuration and attacker inductive bias; permutation-dominant ciphers resist convolutional attackers while remaining vulnerable to permutation-tolerant architectures.

H2 and H3 are the interesting hypotheses. Both are falsifiable, both are informative either way, and both are more sophisticated than the originals.

---

## 7. Literature verification — do this in week one, before anything else

The proposal cites a 2019 work demonstrating a "key-independent deep-learning attack" and a 2025 work applying DL known-plaintext attacks to three chaos-based algorithms. **Verify precisely what "key-independent" meant in each.** Given the analysis in §2, there are three possibilities, and they differ enormously in what they leave open:

1. **The key was fixed but unknown to the attacker** — the network trained on pairs under one secret key. This is the same-key setting. Common, and often loosely described as "without knowing the key". **Leaves your contribution fully intact.**
2. **The key varied at test time and results were reported at the prior floor without acknowledging it** — i.e. plausible-looking outputs, no controls. **Leaves your contribution intact and gives you a correction to make**, which is a strong thesis position.
3. **The key genuinely varied and reconstruction genuinely exceeded chance under proper controls** — then that scheme has key-independent structure they may not have characterised, and your contribution becomes explaining *why*, via the §4.1 knob sweep. **Still intact, different emphasis.**

In all three cases the thesis survives, but the positioning differs, and you need to know which before writing the proposal defence. Retrieve both papers, and specifically check: was the test key in the training set; were controls run; is any reported PSNR distinguishable from a dataset-mean predictor. That last check can often be done from the paper's own figures.

Note this also gives you a possible additional contribution: **the absence of prior-floor controls in the existing neural-cryptanalysis literature is itself a methodological finding**, if it holds up across the papers you review.

---

## 8. Implementation plan

### 8.1 Repository layout

```
Crynexa/
├── crynexa/
│   ├── chaos/
│   │   ├── maps.py          # logistic, tent, henon, pwlcm, arnold — batched
│   │   ├── cipher.py        # ChaosCipher with the §4.1 knobs; AESCTR control
│   │   └── keys.py          # uniform key sampling over the true key space
│   ├── data/
│   │   ├── sources.py       # CIFAR-10 / STL-10 / DIV2K patches / noise control
│   │   └── pairs.py         # on-the-fly (P, C) Dataset; key sampled per item
│   ├── models/
│   │   ├── unet.py          # residual U-Net (primary attacker)
│   │   ├── cnn.py           # plain encoder-decoder baseline
│   │   ├── vit.py           # permutation-tolerant attacker (§3.4)
│   │   └── adapt.py         # few-shot: fine-tune, in-context conditioning
│   ├── metrics/
│   │   ├── recon.py         # MSE, MAE, PSNR, SSIM
│   │   ├── retrieval.py     # top-1/top-k retrieval — the primary metric
│   │   ├── controls.py      # prior floor, mismatch, noise-plaintext
│   │   ├── kgg.py           # normalised Key Generalization Gap
│   │   └── cipher_stats.py  # entropy, correlation, NPCR, UACI
│   ├── train.py
│   ├── eval.py
│   └── sweep.py             # config-matrix runner across the 4 GPUs
├── configs/                 # one YAML per experiment; these are the artifacts
├── reports/                 # auto-generated figures + LaTeX tables
├── tests/
└── docs/
```

### 8.2 Schedule (6 months, adjust to your actual deadline)

**Month 1 — Cipher, data, and the kill test**
- Literature verification (§7). Non-negotiable, week 1.
- Batched chaos maps; `ChaosCipher` with all knobs; AES-CTR control.
- Validate the cipher against published standards: entropy ≈ 7.99, adjacent-pixel correlation ≈ 0, NPCR ≈ 99.6%, UACI ≈ 33.4%. **If your cipher does not meet these, it is a strawman and every subsequent result is worthless.** This validation is a gate, not a step.
- Benchmark on-the-fly generation throughput (§5.3).
- On-the-fly paired dataset; prior-floor and mismatch controls.

> **Run the kill test at the end of month 1.** One config, one U-Net, CIFAR-10, unlimited keys, zero-shot, versus the prior floor. It takes an afternoon once the pipeline exists and it empirically settles the entire feasibility question in §2 before you have invested in anything else. Whatever the answer, you proceed — but you proceed knowing which of your two settings carries the positive results, and you can tell your supervisor exactly where the thesis is heading.

**Month 2 — Zero-shot attack, done properly**
- Residual U-Net + CNN baseline; combined L1 + SSIM loss.
- Same-key baseline (`N_keys = 1`) — must succeed; if it does not, the bug is in your pipeline, not in the science.
- Full `N_keys` sweep → **Curve 1**.
- All four controls (§3.2) reported alongside every number.
- Retrieval metric operational and adopted as primary.

**Month 3 — Cipher-strength sweep**
- Sweep rounds, mode, permutation scope, feedback, precision → **Curve 3**.
- Add the permutation-tolerant architecture; test the §3.4 inductive-bias hypothesis (H3).
- This month produces the thesis's core scientific contribution.

**Month 4 — Few-shot key adaptation**
- Fine-tune baseline → **Curve 2**.
- In-context conditioning model (§4.2b).
- Sample-efficiency comparison: pre-trained vs from-scratch. This is the positive evidence for RQ2 (H2).

**Month 5 — Transfer and scale**
- Cross-map and cross-dataset transfer matrix → **Curve 4**.
- STL-10 and DIV2K; resolution-scaling behaviour.
- Seed replication (≥ 3 seeds) and confidence intervals on every headline number.

**Month 6 — Consolidation**
- Report generator; final figures and tables.
- Thesis writing.
- Optional thin demo UI, if and only if the science is complete.
- Reproducibility pass: fixed seeds, pinned environment, one-command reproduction of every figure.

### 8.3 Deliverables

1. A parameterised chaos-cipher implementation with validated security statistics, plus an AES-CTR control, behind one interface.
2. An on-the-fly plaintext–ciphertext pair generator with correct uniform key sampling.
3. A neural cryptanalysis evaluation protocol with controls and a retrieval-based primary metric — the methodological contribution.
4. Zero-shot and few-shot attack implementations.
5. Four headline figures and the empirical characterisation of which cipher properties confer resistance.
6. A fully reproducible config-driven experimental framework.

---

## 9. Contribution statement for the supervisor

> Existing deep-learning attacks on chaos-based image encryption report plaintext reconstruction from ciphertext, but typically without controlling for the reconstruction attributable to the learned image prior, and without separating same-key from cross-key evaluation. This thesis argues that key-agnostic reconstruction is information-theoretically bounded by the key-independent structure a cipher retains, builds a controlled instrument for measuring that bound, and uses it to identify which structural properties of chaos-based image ciphers — permutation scope, round count, keyed diffusion, plaintext feedback — govern resistance to neural cryptanalysis. It further introduces a few-shot key-adaptive attack setting in which the sample efficiency of adapting to an unseen key provides positive evidence that a model has learned key-independent properties of the cipher family, an effect not observable in the zero-shot setting.

Two claims a supervisor will find credible: a measurement instrument with proper controls, and a mechanism result rather than a demonstration. Both survive negative outcomes.

---

## 10. Summary of changes to the original proposal

| # | Original | Change | Why |
|---|---|---|---|
| 1 | Zero-shot cross-key attack as the primary goal | Reframed as a measurement whose failure is a result; few-shot setting added | Zero-shot is ill-posed for a sound cipher (§2) |
| 2 | 100 enumerated keys, 70/10/20 split | Fresh uniform key per sample; `N_keys` becomes a swept axis | 70 keys cannot cover a 2⁵⁰ key space; enumeration introduces artifacts (§3.1) |
| 3 | No control experiments | Four controls: prior floor, mismatch, AES-CTR, noise plaintexts | Otherwise indistinguishable from learned image prior (§3.2) |
| 4 | PSNR/SSIM primary | Retrieval accuracy primary; PSNR/SSIM secondary | Only retrieval proves *this* image was recovered (§3.3) |
| 5 | NPCR/UACI as attack metrics | Moved to cipher characterisation | They measure the cipher, not the attack (§3.3) |
| 6 | One fixed cipher | Parameterised cipher family with knobs | Converts pass/fail into a dose–response study (§4.1) |
| 7 | U-Net only | U-Net + one permutation-tolerant model | Convolution is the wrong bias for permutation ciphers (§3.4) |
| 8 | `KGG = seen − unseen` | Normalised against the prior floor | Raw gap is uninterpretable without a floor (§4.3) |
| 9 | Attack curve vs training-set size | Curve vs known pairs *under the target key* | The latter is the cryptanalytically meaningful quantity (§4.4) |
| 10 | Web/desktop GUI framework | CLI + YAML configs + report generator | 4–6 weeks for ~0 marks (§5.5) |
| 11 | MSE + SSIM + perceptual loss | L1 + SSIM; perceptual only if needed | Loss engineering is not where the contribution is |
| 12 | Experiments A and B separate | Unified into the `N_keys` axis | One continuous, more informative figure (§3.1) |

Everything else in the original proposal — title, cryptanalysis framing, self-generated dataset, three-tier dataset ladder, implementing the cipher yourself, the cross-map and cross-dataset experiments — stands.
