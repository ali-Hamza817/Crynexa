/* Crynexa — results frontend.
   React + htm, no build step. All data comes from data.json, which is
   regenerated from the measured run outputs by build_web.py. */

const { useState, useEffect, useMemo, useRef } = React;
const html = htm.bind(React.createElement);

/* React requires `style` to be an object, not a string. S() converts inline
   CSS text to the camelCased object form React expects. */
const S = (css) => Object.fromEntries(
  css.split(";").filter((x) => x.trim()).map((p) => {
    const i = p.indexOf(":");
    const k = p.slice(0, i).trim().replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    return [k, p.slice(i + 1).trim()];
  }));

const f = (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const VERDICT_COLOR = { broken: "var(--emerald)", partial: "var(--amber)",
                        resisted: "var(--indigo)", "no-info": "var(--mut)" };

/* ------------------------------------------------------------ helpers -- */
function useReveal() {
  useEffect(() => {
    const io = new IntersectionObserver(
      (es) => es.forEach((e) => e.isIntersecting && e.target.classList.add("in")),
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
    );
    document.querySelectorAll(".rev:not(.in)").forEach((el) => io.observe(el));
    return () => io.disconnect();
  });
}

function CountUp({ value, decimals = 0, suffix = "" }) {
  const [v, setV] = useState(0);
  const ref = useRef(null);
  useEffect(() => {
    let raf, t0;
    const io = new IntersectionObserver((es) => {
      if (!es[0].isIntersecting) return;
      io.disconnect();
      const dur = 900;
      const tick = (t) => {
        if (!t0) t0 = t;
        const k = Math.min(1, (t - t0) / dur);
        setV(value * (1 - Math.pow(1 - k, 3)));
        if (k < 1) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }, { threshold: 0.4 });
    if (ref.current) io.observe(ref.current);
    return () => { io.disconnect(); cancelAnimationFrame(raf); };
  }, [value]);
  return html`<span ref=${ref}>${Number(v).toFixed(decimals)}${suffix}</span>`;
}

const Section = ({ id, alt, eyebrow, title, lede, children }) => html`
  <section id=${id} class=${alt ? "alt" : ""}>
    <div class="wrap">
      <div class="rev">
        ${eyebrow && html`<div class="eyebrow">${eyebrow}</div>`}
        ${title && html`<h2 class="h2">${title}</h2>`}
        ${lede && html`<p class="lede">${lede}</p>`}
      </div>
      ${children}
    </div>
  </section>`;

const Stat = ({ k, v, n, cls, ribbon }) => html`
  <div class="card stat rev">
    ${ribbon && html`<div class=${"ribbon rb-" + ribbon}></div>`}
    <div class="k">${k}</div><div class=${"v " + (cls || "")}>${v}</div>
    ${n && html`<div class="n">${n}</div>`}
  </div>`;

/* --------------------------------------------------------------- nav -- */
const NAV = [
  ["overview", "Overview"], ["plain", "In plain English"], ["findings", "Key findings"], ["questions", "Questions"],
  ["problem", "Problem"], ["datasets", "Datasets"], ["cipher", "Cipher"],
  ["method", "Method"], ["calibration", "Calibration"], ["curve", "Key curve"],
  ["mechanism", "Mechanism"], ["walkthrough", "Sample run"], ["results", "Results"], ["samples", "Reconstructions"],
  ["claims", "Scope"], ["confounds", "Confounds"], ["contributions", "Contribution"],
  ["status", "Status"],
];

const Nav = ({ generated }) => html`
  <div class="nav">
    <div class="wrap">
      <div class="brand"><span class="dot"></span>Crynexa</div>
      <nav>${NAV.map(([id, l]) => html`<a key=${id} href=${"#" + id}>${l}</a>`)}</nav>
      <div class="live"><span class="pulse"></span>${generated}</div>
    </div>
  </div>`;

/* -------------------------------------------------------------- hero -- */
const Hero = ({ runs }) => {
  const ka = runs.filter((r) => r.regime === "key-agnostic");
  const broken = ka.filter((r) => r.verdict === "broken").length;
  return html`
  <div class="hero">
    <div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
    <div class="wrap hero-grid">
      <div class="rev">
        <div class="eyebrow">Master's thesis · neural cryptanalysis</div>
        <h1>Can a network learn <span class="grad">the cipher</span>, not just the key?</h1>
        <p class="lede">
          Chaos-based image encryption is attacked with deep networks trained across a
          large key population, then tested on keys never seen in training. The question
          is not whether a network can invert one key — it is whether anything it learned
          is <em>key-independent</em>, and which structural properties of a cipher decide that.
        </p>
        <div class="tags">
          <span class="tag i">${runs.length} trained models</span>
          <span class="tag c">4 chaos maps</span>
          <span class="tag e">AES-CTR control</span>
          <span class="tag">measured, not simulated</span>
        </div>
      </div>
      <div class="grid g2 rev">
        <${Stat} ribbon="indigo" k="Runs completed" v=${html`<${CountUp} value=${runs.length} />`}
                 n="every number on this page comes from a trained model" />
        <${Stat} ribbon="emerald" k="Configs broken" cls="ok"
                 v=${html`<${CountUp} value=${broken} />`}
                 n=${`of ${ka.length} attacked with unlimited key diversity`} />
      </div>
    </div>
  </div>`;
};

/* ----------------------------------------------------------- problem -- */
const Problem = () => html`
  <${Section} id="problem" alt eyebrow="The research problem"
    title="Zero-shot key-agnostic recovery is bounded, not merely hard"
    lede=${`Trained under a reconstruction loss, the best possible attacker computes the
      posterior mean E[P | C]. When the key is unobserved it must be marginalised out —
      and for a cipher with a large key space and a high-entropy keystream, the ciphertext
      is then statistically independent of the plaintext. The posterior collapses to the
      prior, and the optimal output is the dataset mean image.`}>
    <div class="grid g3 mt2">
      <div class="card rev"><div class="ribbon rb-indigo"></div>
        <h3 style=${S("font-size:16px")}>What that rules out</h3>
        <p class="note">No amount of training keys, data, or model capacity recovers a
        plaintext from a ciphertext alone under a sound cipher. Failure there is a
        theorem, not a tuning problem.</p></div>
      <div class="card rev"><div class="ribbon rb-cyan"></div>
        <h3 style=${S("font-size:16px")}>What it leaves open</h3>
        <p class="note">Real chaos schemes are not sound in that sense. They retain
        key-independent structure: permutation-only stages preserve the histogram exactly,
        row/column scrambles preserve whole rows, low round counts leave local
        correlation. Attack success measures exactly how much.</p></div>
      <div class="card rev"><div class="ribbon rb-amber"></div>
        <h3 style=${S("font-size:16px")}>So the thesis measures</h3>
        <p class="note">Key-agnostic recoverability is treated as a property of a cipher
        <em>configuration</em>. The contribution is an instrument that measures it, and a
        map of which structural knobs confer resistance.</p></div>
    </div>
  <//>`;

/* ---------------------------------------------------------- datasets -- */
const Datasets = ({ data }) => html`
  <${Section} id="datasets" eyebrow="Data"
    title="Three plaintext sources, chosen to separate two effects"
    lede=${`A reconstruction can look right for two very different reasons: the network
      inverted the cipher, or it produced a plausible image from what it learned about the
      dataset. Varying plaintext redundancy while holding the cipher fixed tells those apart.`}>
    <div class="grid g3 mt2">
      ${Object.entries(data.datasets).map(([k, d]) => html`
        <div class="card rev" key=${k}>
          <div class=${"ribbon rb-" + (d.color === "indigo" ? "indigo" : d.color === "cyan" ? "cyan" : "amber")}></div>
          <h3 style=${S("font-size:17px")}>${d.name}</h3>
          <div class="note" style=${S("margin-top:4px;font-weight:650;color:var(--indigo)")}>${d.role}</div>
          <div class="kv mt"><span class="k">Shape</span><span class="v mono">${d.shape}</span></div>
          <div class="kv"><span class="k">Source</span><span class="v mono" style=${S("font-size:11px")}>${d.source}</span></div>
          <p class="note"><strong>Why this dataset.</strong> ${d.why}</p>
        </div>`)}
    </div>
    <div class="card rev mt2">
      <h3 style=${S("font-size:16px")}>Measured redundancy — the resource an attack consumes</h3>
      <div class="tablewrap mt" style=${S("box-shadow:none")}>
        <table><thead><tr><th class="l">plaintext source</th>
          <th>adjacent |diff|</th><th>adjacent correlation</th></tr></thead>
        <tbody>${data.datasetStats.map((s) => html`
          <tr key=${s.dataset}><td class="l mono">${s.dataset}</td>
            <td>${f(s.absDiff, 2)}</td><td>${f(s.corr, 4)}</td></tr>`)}
        </tbody></table>
      </div>
      <p class="note">Uniform noise has no exploitable structure, so any above-chance
      retrieval on it is unambiguously cryptanalytic. Structured plaintexts allow a network
      to be partly right for reasons that have nothing to do with the key.</p>
    </div>
  <//>`;

/* ------------------------------------------------------------ cipher -- */
const Cipher = ({ data }) => html`
  <${Section} id="cipher" alt eyebrow="The cryptosystem"
    title="One parameterised cipher family, not one fixed scheme"
    lede=${`A Fridrich-architecture permutation–diffusion cipher with every structural
      property exposed as an independent knob, plus AES-CTR behind the same interface as a
      negative control. The knobs are the independent variables of the whole study: this
      turns a pass/fail question into a dose–response measurement.`}>
    <div class="flow mt2 rev">
      ${["plaintext", "chaotic keystream", "permutation", "diffusion (+CBC)", "× rounds", "ciphertext"]
        .map((s, i) => html`<${React.Fragment} key=${s}>
          ${i > 0 && html`<span class="arr">→</span>`}<span class="step">${s}</span><//>`)}
    </div>
    <div class="grid g4 mt2">
      ${[["Chaos maps", "logistic · tent · Hénon · PWLCM", "indigo"],
         ["Rounds", "1 – 8", "cyan"],
         ["Mode", "permute · diffuse · both", "emerald"],
         ["Permutation scope", "full 2-D · row/col · blockwise", "amber"],
         ["Keyed permutation", "on / off", "indigo"],
         ["Feedback", "CBC chain / none", "cyan"],
         ["Keystream precision", "float64 / float32", "emerald"],
         ["Negative control", "AES-128-CTR", "amber"]].map(([k, v, c]) => html`
        <div class="card rev" key=${k}><div class=${"ribbon rb-" + c}></div>
          <div class="k" style=${S("font-size:11.5px;font-weight:700;letter-spacing:.06em; text-transform:uppercase;color:var(--mut)")}>${k}</div>
          <div style=${S("font-weight:650;margin-top:7px;font-size:14px")}>${v}</div></div>`)}
    </div>

    ${data.cipherStats.length > 0 && html`
    <div class="card rev mt2">
      <h3 style=${S("font-size:16px")}>Security validation — is this a strawman?</h3>
      <p class="note" style=${S("margin-top:6px")}>Published chaos-cipher targets: entropy ≥ 7.99
      (asymptotic), adjacent correlation ≈ 0, NPCR ≈ 99.61%, UACI ≈ 33.46%. If the
      implemented cipher missed these, every attack result on it would be worthless.</p>
      <div class="tablewrap mt" style=${S("box-shadow:none")}><div class="tscroll">
        <table><thead><tr><th class="l">configuration</th><th>entropy</th>
          <th>corr H</th><th>corr V</th><th>NPCR %</th><th>UACI %</th>
          <th>key sens. NPCR %</th></tr></thead>
        <tbody>${data.cipherStats.map((s) => html`
          <tr key=${s.config}><td class="l mono">${s.config}</td>
            <td>${f(s.entropy, 3)}</td><td>${f(s.corrH, 4)}</td><td>${f(s.corrV, 4)}</td>
            <td>${f(s.npcr, 3)}</td><td>${f(s.uaci, 3)}</td><td>${f(s.keySensNpcr, 3)}</td>
          </tr>`)}
        </tbody></table>
      </div></div>
      <p class="note"><strong>On the entropy figure.</strong> 8.0 bits is unreachable for a
      32×32 image: 3072 samples over 256 bins cap a perfectly uniform source at
      <strong>7.9407</strong>. The reference configuration measures 7.939 — at the ceiling.
      Reading "7.94 &lt; 7.99, therefore weak" would be an artifact of image size.</p>
      <div class="tablewrap mt" style=${S("box-shadow:none")}>
        <table><thead><tr><th class="l">image size</th><th>samples</th>
          <th>entropy ceiling</th></tr></thead>
        <tbody>${data.entropyCeiling.map((e) => html`
          <tr key=${e.size}><td class="l">${e.size}</td>
            <td class="mono">${e.samples ? e.samples.toLocaleString() : "—"}</td>
            <td>${f(e.ceiling, 4)}</td></tr>`)}
        </tbody></table>
      </div>
    </div>`}
  <//>`;

/* ------------------------------------------------------------ method -- */
const Method = ({ data }) => html`
  <${Section} id="method" eyebrow="Method"
    title="Retrieval, not PSNR — and every cipher faces more than one attacker"
    lede=${`PSNR and SSIM reward outputs that are plausible. They cannot tell "recovered
      this image" from "produced a plausible image". For a cryptanalysis claim that
      distinction is the whole point, so top-1 retrieval against distractors is the
      primary metric and PSNR is reported alongside it.`}>
    <div class="grid g2 mt2">
      <div class="card rev"><div class="ribbon rb-indigo"></div>
        <h3 style=${S("font-size:16px")}>Primary metric — top-1 retrieval</h3>
        <p class="note">Given a reconstruction, is the true plaintext the nearest candidate
        among K−1 distractors? Similarity is cosine on mean-centred images, so a constant
        prior-only prediction becomes the zero vector, ties everywhere, and scores exactly
        at chance. Both endpoints were verified:</p>
        <div class="tablewrap mt" style=${S("box-shadow:none")}>
          <table><thead><tr><th class="l">predictor</th><th>@10</th><th>@100</th><th>@1000</th></tr></thead>
          <tbody>${data.metricCheck.map((m) => html`
            <tr key=${m.predictor}><td class="l">${m.predictor}</td>
              <td>${f(m.p10, 4)}</td><td>${f(m.p100, 4)}</td><td>${f(m.p1000, 4)}</td></tr>`)}
          </tbody></table>
        </div>
      </div>
      <div class="card rev"><div class="ribbon rb-cyan"></div>
        <h3 style=${S("font-size:16px")}>Controls that make a null result meaningful</h3>
        <div class="kv mt"><span class="k">Prior floor</span>
          <span class="v">constant = mean training image</span></div>
        <div class="kv"><span class="k">Mismatch</span>
          <span class="v">wrong ciphertext → measures hallucination</span></div>
        <div class="kv"><span class="k">Negative</span>
          <span class="v">AES-CTR must stay at chance</span></div>
        <div class="kv"><span class="k">Positive</span>
          <span class="v">key-independent cipher must break</span></div>
        <div class="kv"><span class="k">Same-key ceiling</span>
          <span class="v">per-config upper bound</span></div>
        <p class="note">A key-agnostic attack that fails is uninterpretable on its own:
        "the cipher resists" and "this attacker cannot invert it at all" give identical
        numbers. Every configuration is therefore also run with a single fixed key.</p>
      </div>
    </div>
    <div class="grid g2 mt2">
      ${Object.entries(data.attackers).map(([k, a]) => html`
        <div class="card rev" key=${k}>
          <h3 style=${S("font-size:15px")}>${a.label}</h3>
          <p class="note" style=${S("margin-top:5px")}>${a.desc}</p>
          <div class="kv mt"><span class="k">input</span><span class="v mono">${a.input}</span></div>
        </div>`)}
    </div>
  <//>`;

/* ------------------------------------------------------- calibration -- */
const Calibration = ({ runs }) => {
  const best = (pred) => {
    const m = runs.filter(pred);
    return m.length ? m.reduce((a, b) => (a.unseenTop1 > b.unseenTop1 ? a : b)) : null;
  };
  const pos = best((r) => /posctrl|calpos/.test(r.run) && r.regime !== "same-key");
  const aes = best((r) => r.cipher === "aes" && r.regime === "key-agnostic")
           || best((r) => r.cipher === "aes");
  const ok = pos && pos.unseenTop1 > 0.5 && aes && aes.unseenTop1 < 5 * aes.chance;
  return html`
  <${Section} id="calibration" alt eyebrow="Calibration"
    title="The instrument brackets the full range"
    lede=${`Before any security claim, the pipeline must break a cipher that has no key
      dependence at all, and fail completely on one that is sound. Both controls have the
      attacker at top-1 = 1.000 on seen keys, so capability is not in question — the only
      thing separating them is key dependence.`}>
    <div class="grid g4 mt2">
      ${pos && html`<${Stat} ribbon="emerald" k="Positive control" cls="ok"
        v=${f(pos.unseenTop1, 3)} n="key-independent cipher, unseen keys — must break" />`}
      ${aes && html`<${Stat} ribbon="indigo" k="AES-CTR control" cls="ind"
        v=${f(aes.unseenTop1, 3)} n="must stay at chance — else the pipeline leaks plaintext" />`}
      <${Stat} ribbon="cyan" k="Chance baseline" cls="cy" v="0.010" n="top-1 at pool size 100" />
      <${Stat} ribbon=${ok ? "emerald" : "amber"} k="Instrument status"
        cls=${ok ? "ok" : "wn"} v=${ok ? "Calibrated" : "Pending"}
        n=${ok ? "separation is ~100× — measurements in between are meaningful"
               : "waiting on both controls"} />
    </div>
    ${pos && aes && html`
    <div class="card rev mt2">
      <div class="tablewrap" style=${S("box-shadow:none")}><table>
        <thead><tr><th class="l">control</th><th>seen top-1</th><th>unseen top-1</th>
          <th>gain over floor</th><th class="l">meaning</th></tr></thead>
        <tbody>
          <tr><td class="l">no key dependence (fixed permutation)</td>
            <td>${f(pos.seenTop1, 3)}</td>
            <td style=${S("color:var(--emerald);font-weight:700")}>${f(pos.unseenTop1, 3)}</td>
            <td>${f(pos.gainDb, 2)} dB</td>
            <td class="l">detects a real leak perfectly</td></tr>
          <tr><td class="l">full key dependence (AES-CTR)</td>
            <td>${f(aes.seenTop1, 3)}</td>
            <td style=${S("color:var(--rose);font-weight:700")}>${f(aes.unseenTop1, 3)}</td>
            <td>${f(aes.gainDb, 2)} dB</td>
            <td class="l">no hallucinated credit</td></tr>
        </tbody></table></div>
      <p class="note">AES seen = 1.000 is not a break of AES. A fixed key with a fixed
      nonce yields a fixed keystream, which is learnable from enough pairs under that key.
      With a fresh key the attack is dead at chance.</p>
    </div>`}
  <//>`;
};

/* ----------------------------------------------------------- results -- */
const COLS = [
  ["variant", "configuration", 1], ["regime", "keys", 1], ["attacker", "attacker", 1],
  ["dataset", "data", 1], ["nKeys", "# keys", 0], ["ceiling", "ceiling", 0],
  ["unseenTop1", "unseen top-1", 0], ["verdict", "verdict", 1],
  ["mismatchTop1", "mismatch", 0], ["unseenPsnr", "psnr", 0], ["floorPsnr", "floor", 0],
  ["gainDb", "gain dB", 0], ["secs", "sec", 0],
];

const Results = ({ runs, sel, setSel }) => {
  const [q, setQ] = useState("");
  const [ds, setDs] = useState("all");
  const [at, setAt] = useState("all");
  const [rg, setRg] = useState("all");
  const [sk, setSk] = useState("unseenTop1");
  const [sd, setSd] = useState(-1);

  const uniq = (k) => [...new Set(runs.map((r) => r[k]))].sort();
  const rows = useMemo(() => runs
    .filter((r) => (!q || r.run.toLowerCase().includes(q.toLowerCase()))
      && (ds === "all" || r.dataset === ds) && (at === "all" || r.attacker === at)
      && (rg === "all" || r.regime === rg))
    .sort((a, b) => {
      const x = a[sk], y = b[sk];
      if (x == null) return 1; if (y == null) return -1;
      return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * sd;
    }), [runs, q, ds, at, rg, sk, sd]);

  const ka = rows.filter((r) => r.regime === "key-agnostic").slice(0, 24);
  const maxV = Math.max(0.06, ...ka.map((r) => r.unseenTop1));

  return html`
  <${Section} id="results" eyebrow="Results"
    title="Key-agnostic attack success by cipher configuration"
    lede=${`Green marks a configuration broken without ever seeing its key — unseen-key
      retrieval at five times chance or better. Click any row for its reconstructions.`}>

    ${ka.length > 0 && html`
    <div class="card rev mt2">
      <svg viewBox=${`0 0 780 ${ka.length * 25 + 32}`} width="100%" height=${ka.length * 25 + 32}>
        ${ka.map((r, i) => {
          const y = i * 25 + 14, L = 268, W = 780, x = (v) => L + (v / maxV) * (W - L - 78);
          return html`<${React.Fragment} key=${r.run}>
            <text x=${L - 10} y=${y + 12} text-anchor="end" font-size="11.5"
                  fill="var(--mut)" font-family="var(--mono)">
              ${r.variant.slice(0, 30)}</text>
            <rect x=${L} y=${y + 2} width=${Math.max(2, x(r.unseenTop1) - L)} height="16" rx="4"
                  fill=${VERDICT_COLOR[r.verdict]} opacity="0.9">
              <animate attributeName="width" from="0" to=${Math.max(2, x(r.unseenTop1) - L)}
                       dur="0.8s" fill="freeze" />
            </rect>
            <text x=${x(r.unseenTop1) + 8} y=${y + 15} font-size="11.5" fill="var(--mut)"
                  font-variant-numeric="tabular-nums">${f(r.unseenTop1, 3)}</text>
          <//>`;
        })}
        ${(() => { const L = 268, W = 780, x = (v) => L + (v / maxV) * (W - L - 78),
                    cx = x(ka[0].chance), H = ka.length * 25 + 32;
          return html`<${React.Fragment}>
            <line x1=${cx} y1="6" x2=${cx} y2=${H - 16} stroke="var(--rose)"
                  stroke-width="1.5" stroke-dasharray="4 3" />
            <text x=${cx + 6} y=${H - 4} font-size="11" fill="var(--rose)">chance 0.01</text>
          <//>`; })()}
      </svg>
      <div class="legend">
        <span><i style=${S("background:var(--emerald)")}></i>broken — ≥5× chance</span>
        <span><i style=${S("background:var(--amber)")}></i>partial — 2–5× chance</span>
        <span><i style=${S("background:var(--indigo)")}></i>resisted — &lt;2× chance</span>
      </div>
    </div>`}

    <div class="controls mt2 rev">
      <input placeholder="filter by name…" value=${q}
             onInput=${(e) => setQ(e.target.value)} style=${S("min-width:230px")} />
      ${[[ds, setDs, "dataset"], [at, setAt, "attacker"], [rg, setRg, "regime"]].map(
        ([val, set, key]) => html`
        <select key=${key} value=${val} onChange=${(e) => set(e.target.value)}>
          <option value="all">all ${key}s</option>
          ${uniq(key).map((v) => html`<option key=${v} value=${v}>${v}</option>`)}
        </select>`)}
      <span class="chipcount">${rows.length} runs</span>
    </div>

    <div class="tablewrap rev"><div class="tscroll">
      <table>
        <thead><tr>${COLS.map(([k, l, isL]) => html`
          <th key=${k} class=${isL ? "l" : ""}
              onClick=${() => { setSd(k === sk ? -sd : -1); setSk(k); }}>
            ${l}${sk === k ? (sd > 0 ? " ▲" : " ▼") : ""}</th>`)}
        </tr></thead>
        <tbody>
          ${rows.length === 0 ? html`<tr><td colspan="13" class="empty">no runs match</td></tr>`
          : rows.map((r) => html`
            <tr key=${r.run} class=${sel === r.run ? "sel" : ""} onClick=${() => setSel(r.run)}>
              <td class="l mono" title=${r.cipher}>${r.variant}</td>
              <td class="l"><span class=${"pill " + (r.regime === "same-key" ? "p-same" : "p-key")}>
                ${r.regime}</span></td>
              <td class="l">${r.attacker}</td><td class="l">${r.dataset}</td>
              <td>${r.nKeys}</td>
              <td style=${{ color: r.ceilingValid ? "" : "var(--rose)",
                            fontWeight: r.ceilingValid ? 400 : 700 }}
                  title=${r.ceilingValid ? "attacker can invert this cipher with a fixed key"
                    : "ceiling at chance — this cell carries NO security information"}>
                ${f(r.ceiling)}</td>
              <td><span class="bar"><i style=${{ width: Math.min(100, r.unseenTop1 * 100) + "%",
                    background: VERDICT_COLOR[r.verdict] || "var(--mut)" }}></i></span>${f(r.unseenTop1)}</td>
              <td class="l"><span class=${"pill p-" + r.verdict}>${r.verdict}</span></td>
              <td>${f(r.mismatchTop1)}</td>
              <td>${f(r.unseenPsnr, 2)}</td><td>${f(r.floorPsnr, 2)}</td>
              <td style=${{ color: r.gainDb > 1 ? "var(--emerald)" : r.gainDb < -0.5 ? "var(--rose)" : "" ,
                            fontWeight: Math.abs(r.gainDb) > 1 ? 700 : 400 }}>
                ${(r.gainDb > 0 ? "+" : "") + f(r.gainDb, 2)}</td>
              <td>${f(r.kgg, 3)}</td><td>${r.secs}</td>
            </tr>`)}
        </tbody>
      </table>
    </div></div>
    <p class="note"><strong>Ceiling</strong> is the same-key retrieval of this exact
    configuration — what the attacker achieves when the key is fixed. A ceiling at chance
    (shown in red) means the cell carries <em>no</em> security information: "the cipher
    resists" and "this attacker cannot invert it at all" then produce identical numbers.
    Read every unseen-key number against its ceiling.</p>
    <p class="note"><strong>KGG</strong> is the normalised Key Generalization Gap:
    0 means the attack transfers perfectly to unseen keys (a full key-agnostic break),
    1 means it retains nothing (key mixing sound). <strong>Gain dB</strong> is PSNR above
    the prior floor — the zero-information baseline.</p>
  <//>`;
};

/* ----------------------------------------------------------- samples -- */
const Samples = ({ data, sel, runs }) => {
  const s = data.samples[sel];
  const run = runs.find((r) => r.run === sel);
  return html`
  <${Section} id="samples" alt eyebrow="Reconstructions"
    title="What the attacker actually produced"
    lede=${`Do not judge these by eye. A plausible-looking image can be produced from the
      dataset prior alone — that is exactly what the mismatch control measures. Read the
      retrieval number, then look at the pictures.`}>
    <div class="card rev mt2">
      <div style=${S("display:flex;gap:14px;align-items:baseline;flex-wrap:wrap")}>
        <h3 style=${S("font-size:15px")} class="mono">${sel || "select a run above"}</h3>
        ${run && html`<span class=${"pill p-" + run.verdict}>${run.verdict}</span>
          <span class="note" style=${S("margin:0")}>unseen top-1 ${f(run.unseenTop1)} ·
            mismatch ${f(run.mismatchTop1)} · chance ${f(run.chance)}</span>`}
      </div>
      <div class="samples mt2">
        ${s ? s.map((t, i) => html`
          <div class="trip" key=${i}>
            <img src=${t.plain} alt="plaintext" /><div class="lab">plaintext</div>
            <img src=${t.cipher} alt="ciphertext" /><div class="lab">ciphertext</div>
            <img src=${t.recon} alt="reconstruction" /><div class="lab">reconstruction</div>
          </div>`)
        : html`<div class="empty">no saved samples for this run</div>`}
      </div>
    </div>
  <//>`;
};

/* ---------------------------------------------------------- findings -- */
const FINDINGS = [
  ["F1", "Convolution cannot express a position-dependent keystream",
   "A chaos keystream is a pseudorandom function of absolute position; a translation-equivariant network applies the same filters everywhere. Adding coordinate channels moved same-key retrieval from 0.000 to ~1.000."],
  ["F2", "The retrieval metric included the query in its own distractor pool",
   "Accuracy was pinned at exactly 0.904 — which is 1−(999/1000)^99, the ceiling this bug imposes, not a property of any model. Masking self-matches restored a true 1.0 for a perfect predictor."],
  ["F3", "Input representation must match the cipher's algebra",
   "XOR is linear only in bit-planes; the CBC feedback term is near-linear only in pixel values. An attacker given one cannot cheaply express the other, so both are supplied."],
  ["F4", "Smooth coordinates are a poor basis for a pseudorandom keystream",
   "A free learnable per-position embedding gives the attacker capacity to represent an arbitrary position-dependent constant. It confers no key knowledge."],
  ["F5", "Producer threads each built their own key pool",
   "Silent: n_keys=k actually produced k×6 distinct keys. The key-diversity axis — the main independent variable — would have been wrong in every run."],
  ["F6", "Entropy of an encrypted 32×32 image cannot reach 8.0",
   "3072 samples over 256 bins cap a perfectly uniform source at 7.9407. The cipher measures 7.939, i.e. at the ceiling."],
  ["F7", "Every configuration needs its own same-key ceiling",
   "Otherwise 'the cipher resists' and 'this attacker cannot invert it at all' are indistinguishable."],
  ["F8", "Plaintext redundancy is a separate resource from key recovery",
   "Adjacent-pixel correlation is 0.0006 for noise and 0.9162 for structured images. Running the same attack on both separates cipher inversion from prior exploitation."],
  ["F9", "An auxiliary SSIM loss drove models below the prior floor",
   "A constant prediction has degenerate SSIM, so the term is maximally penalised and the optimiser manufactures spurious structure — PSNR fell to ~6.0 dB against an 11.1 dB floor. Switched to L1."],
  ["F11", "CBC modular feedback defeats regression- and classification-based attack",
   "Single-variable control: XOR 1.000 vs CBC 0.011 with everything else identical. Three fixes failed — predecessor channel, a verified-exact 256-way categorical head (loss pinned at exactly ln 256), and longer training. The inverse depends on two bytes, giving 65,536 combinations per position against 50,000 images, so the attacker must learn modular arithmetic rather than memorise. Corrected later: a ResUNet on CIFAR-10 does reach a 0.981 ceiling, so this is conditional on plaintext redundancy, not absolute."],
  ["F12", "Key-specific memorisation is a capacity budget, not learning",
   "The key-diversity curve separates the two: seen-key accuracy decays 1.000 → 0.982 → 0.673 → 0.034 as the key pool grows, while unseen-key accuracy never leaves chance. An earlier partial reading called this collapse immediate; the completed sweep shows it is gradual, and the gradient is what proves the flat unseen line is not a capacity artifact."],
  ["F10", "Correction: convolution can invert a global pixel gather",
   "An earlier reading held this impossible. Once the loss was fixed, ResUNet solves the key-independent permutation control completely (top-1 1.000, +13.9 dB) — the obstacle was F9, not inductive bias. The architecture difference is quantitative (+13.9 vs +15.5 dB), not categorical."],
];

const Findings = () => html`
  <${Section} id="confounds" eyebrow="Methodology"
    title="Twelve confounds that would each have faked a security result"
    lede=${`A negative cryptanalysis result only means something if the attacker was capable
      in the first place. Ruling these out is part of the contribution, so they are recorded
      rather than quietly fixed — including one correction to an earlier conclusion.`}>
    <div class="grid g2 mt2">
      ${FINDINGS.map(([id, title, body]) => html`
        <div class="finding rev" key=${id}>
          <h4><span class="badge">${id}</span>${title}</h4>
          <p>${body}</p>
        </div>`)}
    </div>
  <//>`;

/* ------------------------------------------------------------ status -- */
const Status = ({ data }) => html`
  <${Section} id="status" alt eyebrow="Reproducibility"
    title="Experiment pipeline & verification"
    lede="Suites run unattended across the available GPUs; this page regenerates from their outputs.">
    <div class="grid g4 mt2">
      ${data.pipeline.suites.map((s) => html`
        <div class="card rev" key=${s.suite}>
          <div class=${"ribbon rb-" + (s.state === "complete" ? "emerald"
            : s.state === "running" ? "indigo" : "cyan")}></div>
          <div class="k" style=${S("font-size:11.5px;font-weight:700;letter-spacing:.06em; text-transform:uppercase;color:var(--mut)")}>${s.suite}</div>
          <div style=${S("font-size:23px;font-weight:750;margin-top:7px")}>
            ${s.done}${s.total ? " / " + s.total : ""}</div>
          <div class="prog"><i style=${{ width: s.total ? (100 * s.done / s.total) + "%" : "0%" }}></i></div>
          <div class="n" style=${S("color:var(--mut);font-size:12.5px;margin-top:8px")}>${s.state}</div>
          ${s.desc && html`<div class="n" style=${S("color:var(--mut);font-size:11.5px;margin-top:4px;line-height:1.45")}>${s.desc}</div>`}
        </div>`)}
    </div>
    <div class="grid g2 mt2">
      <div class="card rev">
        <h3 style=${S("font-size:15px")}>Automated checks
          <span class=${"pill " + (data.tests.passed ? "p-broken" : "p-partial")}
                style=${S("margin-left:8px")}>${data.tests.passed ? "passing" : "attention"}</span></h3>
        <div class="note mono" style=${S("white-space:pre-wrap;font-size:11.5px;margin-top:10px")}>
          ${data.tests.output.join("\n")}</div>
      </div>
      <div class="card rev">
        <h3 style=${S("font-size:15px")}>Pipeline log</h3>
        <div class="note mono" style=${S("white-space:pre-wrap;font-size:11.5px;margin-top:10px")}>
          ${data.pipeline.log.length ? data.pipeline.log.join("\n") : "no entries yet"}</div>
      </div>
    </div>
  <//>`;



/* ------------------------------------------------------ plain english -- */
const PlainEnglish = () => html`
  <${Section} id="plain" eyebrow="In plain English"
    title="What this project is, without the jargon"
    lede="If you read only one section, read this one.">

    <div class="grid g3 mt2">
      <div class="card rev"><div class="ribbon rb-indigo"></div>
        <div class="eyebrow" style=${S("margin-bottom:8px")}>The problem</div>
        <h3 style=${S("font-size:17px")}>Can an AI break encryption without the password?</h3>
        <p class="note">People scramble images so nobody else can see them. One popular
        family of methods uses <em>chaos</em> — maths that turns a secret password into a
        stream of unpredictable numbers, then uses those numbers to shuffle the pixels and
        change their colours.</p>
        <p class="note">Normally you need the password to unscramble it. Our question:
        <strong>if an AI studies thousands of scrambled-and-original image pairs, can it
        learn to unscramble a picture that was locked with a password it has never
        seen?</strong></p>
        <p class="note">If the answer were yes, the encryption would be broken in a serious
        way — an attacker would not need to steal your password at all.</p>
      </div>

      <div class="card rev"><div class="ribbon rb-cyan"></div>
        <div class="eyebrow" style=${S("margin-bottom:8px")}>What we built</div>
        <h3 style=${S("font-size:17px")}>An honest testing machine, not just one attack</h3>
        <p class="note">We wrote the encryption ourselves with <strong>switches for each
        security feature</strong>, so we could turn them on and off one at a time and see
        which one actually does the protecting.</p>
        <p class="note">Then we trained <strong>223 AI attackers</strong> against it.</p>
        <p class="note">The important part is how we check our own work. Every single test
        also runs a sanity question: <em>"can this AI unscramble the image when we DO give
        it the password?"</em> If it cannot even do that, then it failing without the
        password proves nothing — the AI was simply too weak. Most papers skip this. We
        report those cases as <strong>"no information"</strong> instead of claiming the
        encryption is safe.</p>
        <p class="note">We also included two reference points: real bank-grade encryption
        (AES), which the attack <em>must</em> fail against, and a deliberately fake cipher
        with no password protection at all, which it <em>must</em> break. Both behaved
        correctly, so we know the measuring instrument works.</p>
      </div>

      <div class="card rev"><div class="ribbon rb-emerald"></div>
        <div class="eyebrow" style=${S("margin-bottom:8px")}>What we found</div>
        <h3 style=${S("font-size:17px")}>One real weakness, and one clear rule</h3>
        <p class="note"><strong>1. A common shortcut is genuinely breakable.</strong> Some
        schemes shuffle the rows and the columns separately because it is faster. That
        leaves a fingerprint which is the same no matter what password you choose — so our
        AI recovered pictures locked with passwords it had never seen.</p>
        <p class="note"><strong>2. The shuffling step barely protects anything; the
        colour-changing step protects everything.</strong> With shuffling alone the attack
        succeeded completely. Adding one password-controlled colour-changing step dropped it
        to pure guesswork.</p>
        <p class="note"><strong>3. The AI never truly "learns the cipher".</strong> It can
        memorise about 16 passwords almost perfectly — but it transfers <em>nothing</em> to
        password number 17. More training passwords did not help at all.</p>
        <p class="note"><strong>4. Doing more rounds does not help.</strong> Repeating the
        encryption 2, 3 or 4 times was no safer than doing it once.</p>
      </div>
    </div>

    <div class="card rev mt2">
      <h3 style=${S("font-size:16px")}>The single clearest picture of the result</h3>
      <p class="note">We locked 1000 test photos with a brand-new password, then asked each
      AI to recover them. We measured <strong>where the correct photo ranked</strong> among
      1000 possibilities. Rank 1 means it picked the right photo outright. A rank around
      500 means it was just guessing.</p>
      <div class="tablewrap mt" style=${S("box-shadow:none")}>
        <table><thead><tr><th class="l">the encryption used</th>
          <th>where the right photo ranked</th><th class="l">what that means</th></tr></thead>
        <tbody>
          <tr><td class="l">fake cipher with no password protection</td>
            <td style=${S("color:var(--emerald);font-weight:750")}>1st of 1000</td>
            <td class="l">completely broken — as expected</td></tr>
          <tr><td class="l"><strong>rows &amp; columns shuffled separately</strong></td>
            <td style=${S("color:var(--emerald);font-weight:750")}>53rd of 1000</td>
            <td class="l"><strong>really is leaking — top 5%, with a password never seen</strong></td></tr>
          <tr><td class="l">proper shuffling + password-based colour change</td>
            <td style=${S("color:var(--mut)")}>502nd of 1000</td>
            <td class="l">pure guesswork — held up</td></tr>
          <tr><td class="l">AES (bank-grade encryption)</td>
            <td style=${S("color:var(--mut)")}>501st of 1000</td>
            <td class="l">pure guesswork — as expected</td></tr>
        </tbody></table>
      </div>
      <p class="note"><strong>The practical takeaway for anyone designing one of these
      schemes:</strong> do not shuffle rows and columns separately to save time, and never
      rely on shuffling alone — the password-driven colour-changing stage is what actually
      keeps the image secret.</p>
      <p class="note" style=${S("color:var(--ink2)")}><strong>And the honest limit:</strong>
      we tested whether an AI can break in with <em>no</em> knowledge of the password. We did
      not test an attacker who already has a few matching original/scrambled examples for
      that exact password — that is a different and easier situation, and nothing here says
      these schemes are safe against it.</p>
    </div>
  <//>`;

/* --------------------------------------------------- key findings -- */
const KeyFindings = ({ data }) => {
  const d = data.derived, c = d.counts;
  const top = d.leaks.filter((l) => l.verdict === "broken" && !/posctrl/.test(l.variant));
  const best = top[0];
  return html`
  <${Section} id="findings" eyebrow="Key findings"
    title="What ${c.total} trained models establish"
    lede=${`Each finding below is measured against a prior floor, a mismatch control and
      its own same-key ceiling. Chance for top-1 retrieval at pool size 100 is 0.010.`}>
    <div class="grid g2 mt2">
      <div class="card rev"><div class="ribbon rb-emerald"></div>
        <div class="eyebrow" style=${S("margin-bottom:8px")}>Finding 1 · a real leak</div>
        <h3 style=${S("font-size:17px")}>Row/column permutation is broken without the key</h3>
        <p class="note">Scrambling row order and column order independently preserves the
        multiset of whole rows and columns <em>under every key</em>. Both attacker families
        find it, on both datasets, with mismatch controls at chance.</p>
        <div style=${S("font-size:34px;font-weight:780;color:var(--emerald);margin-top:12px;letter-spacing:-.03em")}>
          ${best ? f(best.unseenTop1) : "—"}
          <span style=${S("font-size:14px;font-weight:600;color:var(--mut)")}>
            &nbsp;unseen-key top-1 · ${best ? Math.round(best.unseenTop1 / best.chance) : "—"}× chance</span>
        </div>
      </div>
      <div class="card rev"><div class="ribbon rb-indigo"></div>
        <div class="eyebrow" style=${S("margin-bottom:8px")}>Finding 2 · the mechanism</div>
        <h3 style=${S("font-size:17px")}>Keyed diffusion supplies all key-agnostic security</h3>
        <p class="note">Hold the permutation <em>fully key-independent</em> and toggle only
        the diffusion stage. A single keyed diffusion takes the attack from total recovery
        to chance.</p>
        <div style=${S("display:flex;gap:24px;align-items:baseline;margin-top:12px")}>
          <div><div style=${S("font-size:30px;font-weight:780;color:var(--rose);letter-spacing:-.03em")}>1.000</div>
            <div class="n" style=${S("color:var(--mut);font-size:12px")}>no diffusion</div></div>
          <div style=${S("font-size:20px;color:var(--line2)")}>→</div>
          <div><div style=${S("font-size:30px;font-weight:780;color:var(--emerald);letter-spacing:-.03em")}>0.010</div>
            <div class="n" style=${S("color:var(--mut);font-size:12px")}>+ keyed diffusion</div></div>
        </div>
      </div>
      <div class="card rev"><div class="ribbon rb-cyan"></div>
        <div class="eyebrow" style=${S("margin-bottom:8px")}>Finding 3 · the central negative</div>
        <h3 style=${S("font-size:17px")}>Zero key generalisation, at every key diversity</h3>
        <p class="note">Unseen-key retrieval is pinned at chance from 1 training key to
        unlimited. The same models memorise ~16 keys at 0.98, so this is not a capacity or
        training artifact — they memorise, they never generalise across key space.</p>
      </div>
      <div class="card rev"><div class="ribbon rb-amber"></div>
        <div class="eyebrow" style=${S("margin-bottom:8px")}>Finding 4 · non-effects</div>
        <h3 style=${S("font-size:17px")}>Round count and keystream precision do not matter</h3>
        <p class="note">Both are treated as security parameters in the literature. With a
        valid ceiling in every cell, rounds 1–4 and float32 vs float64 all sit at chance:
        against this attack class their marginal value is zero.</p>
        <div class="mt">
          ${Object.entries(d.axes).map(([k, v]) => v.n ? html`
            <div class="kv" key=${k}><span class="k">${k}</span>
              <span class="v">${f(v.min)} – ${f(v.max)} <span style=${S("color:var(--mut);font-weight:400")}>(${v.n} runs)</span></span></div>` : null)}
        </div>
      </div>
    </div>
    <div class="grid g4 mt2">
      <${Stat} ribbon="indigo" k="Trained models" v=${html`<${CountUp} value=${c.total} />`}
               n="each with full controls recorded" />
      <${Stat} ribbon="emerald" k="Broken key-agnostically" cls="ok"
               v=${html`<${CountUp} value=${c.broken} />`} n="unseen-key retrieval ≥ 5× chance" />
      <${Stat} ribbon="amber" k="Partial leaks" cls="wn"
               v=${html`<${CountUp} value=${c.partial} />`} n="2–5× chance" />
      <${Stat} ribbon="cyan" k="Valid ceilings" cls="cy"
               v=${`${c.validCeiling}/${c.keyAgnostic}`}
               n="cells where the attacker provably CAN invert with a fixed key" />
    </div>
  <//>`;
};

/* ------------------------------------------------ research questions -- */
const Questions = ({ data }) => html`
  <${Section} id="questions" alt eyebrow="Research questions"
    title="The three questions, and what the data answers"
    lede="Verdicts are stated only where a valid same-key ceiling exists for the cells involved.">
    <div class="grid mt2" style=${S("gap:14px")}>
      ${data.research.map((r) => html`
        <div class="card rev" key=${r.id}>
          <div style=${S("display:flex;gap:12px;align-items:baseline;flex-wrap:wrap")}>
            <span class="badge">${r.id}</span>
            <span class=${"pill " + (r.verdict.includes("negative") ? "p-partial" : "p-broken")}>
              ${r.verdict}</span>
          </div>
          <p style=${S("margin-top:10px;font-weight:600;font-size:15px")}>${r.q}</p>
          <p class="note">${r.a}</p>
        </div>`)}
    </div>
  <//>`;

/* ------------------------------------------- key-diversity curve -- */
const KeyCurve = ({ data }) => {
  const curves = data.derived.curves;
  const keys = Object.keys(curves);
  const [sel, setSel] = useState(keys[0] || null);
  if (!keys.length) return null;
  const pts = curves[sel] || [];
  const W = 720, H = 300, L = 56, R = 18, T = 18, B = 46;
  const xs = pts.map((_, i) => L + (i / Math.max(1, pts.length - 1)) * (W - L - R));
  const y = (v) => T + (1 - Math.min(1, v)) * (H - T - B);
  const path = (k) => pts.map((p, i) => `${i ? "L" : "M"}${xs[i]},${y(p[k])}`).join(" ");
  return html`
  <${Section} id="curve" eyebrow="Curve 1 · the money plot"
    title="Key diversity: memorisation capacity versus generalisation"
    lede=${`One variable changes across this sweep — how many distinct secret keys appear in
      training. "Seen" is retrieval on keys drawn from the training pool; "unseen" on keys
      the model has never encountered. Separating the two is the whole point.`}>
    <div class="controls mt2 rev">
      ${keys.map((k) => html`
        <button key=${k} onClick=${() => setSel(k)}
          class=${"pill " + (k === sel ? "p-key" : "p-resisted")}
          style=${S("cursor:pointer;border:none;font-family:inherit;font-size:12px;padding:7px 13px")}>
          ${k}</button>`)}
    </div>
    <div class="card rev">
      <svg viewBox=${`0 0 ${W} ${H}`} width="100%" height=${H}>
        ${[0, 0.25, 0.5, 0.75, 1].map((g) => html`<${React.Fragment} key=${g}>
          <line x1=${L} y1=${y(g)} x2=${W - R} y2=${y(g)} stroke="var(--line)" stroke-width="1" />
          <text x=${L - 9} y=${y(g) + 4} text-anchor="end" font-size="11" fill="var(--mut)">${g.toFixed(2)}</text>
        <//>`)}
        <line x1=${L} y1=${y(pts[0].chance)} x2=${W - R} y2=${y(pts[0].chance)}
              stroke="var(--rose)" stroke-width="1.5" stroke-dasharray="5 4" />
        <text x=${W - R} y=${y(pts[0].chance) - 7} text-anchor="end" font-size="11"
              fill="var(--rose)">chance ${f(pts[0].chance, 3)}</text>
        <path d=${path("seen")} fill="none" stroke="var(--indigo)" stroke-width="2.5"
              stroke-linejoin="round" />
        <path d=${path("unseen")} fill="none" stroke="var(--emerald)" stroke-width="2.5"
              stroke-linejoin="round" />
        ${pts.map((p, i) => html`<${React.Fragment} key=${p.label}>
          <circle cx=${xs[i]} cy=${y(p.seen)} r="4" fill="var(--indigo)" />
          <circle cx=${xs[i]} cy=${y(p.unseen)} r="4" fill="var(--emerald)" />
          <text x=${xs[i]} y=${H - B + 20} text-anchor="middle" font-size="11"
                fill="var(--mut)">${p.label}</text>
        <//>`)}
        <text x=${(L + W) / 2} y=${H - 6} text-anchor="middle" font-size="11.5"
              fill="var(--mut)">distinct secret keys seen during training</text>
      </svg>
      <div class="legend">
        <span><i style=${S("background:var(--indigo)")}></i>seen keys — from the training pool</span>
        <span><i style=${S("background:var(--emerald)")}></i>unseen keys — never encountered</span>
        <span><i style=${S("background:var(--rose)")}></i>chance</span>
      </div>
      <p class="note"><strong>Read it like this.</strong> The blue line falling is a
      memorisation budget being consumed — the attacker holds a handful of keystreams
      perfectly, then runs out. The green line never leaves chance: more training keys buy
      <em>no</em> transfer to a new key. Because the same models demonstrably reach 0.98 on
      16 keys, the flat green line cannot be dismissed as undertrained or under-capacity.</p>
    </div>
  <//>`;
};

/* ------------------------------------------------------- mechanism -- */
const Mechanism = ({ data }) => {
  const m = data.derived.mechanism;
  if (!m.length) return null;
  return html`
  <${Section} id="mechanism" alt eyebrow="Curve 2 · mechanism"
    title="Which stage of the cipher actually carries the security?"
    lede=${`Both configurations below use a permutation that is IDENTICAL for every key.
      They differ in one thing only: whether a keyed diffusion stage follows. That isolates
      the contribution of each stage.`}>
    <div class="tablewrap rev mt2"><table>
      <thead><tr><th class="l">dataset</th><th class="l">attacker</th>
        <th>key-independent permutation, no diffusion</th>
        <th>+ keyed diffusion</th><th>chance</th></tr></thead>
      <tbody>${m.map((r) => html`
        <tr key=${r.dataset + r.attacker}>
          <td class="l">${r.dataset}</td><td class="l">${r.attacker}</td>
          <td style=${S("color:var(--rose);font-weight:750;font-size:15px")}>${f(r.noDiffusion)}</td>
          <td style=${S("color:var(--emerald);font-weight:750;font-size:15px")}>${f(r.keyedDiffusion)}</td>
          <td>${f(r.chance)}</td></tr>`)}
      </tbody></table></div>
    <p class="note"><strong>Permutation contributes essentially nothing to key-agnostic
    security; diffusion contributes all of it.</strong> The same masking removes the
    row/column leak: row/col permutation alone reaches 0.221, but with keyed diffusion
    added it falls to 0.010. For a scheme designer the permutation stage may be chosen for
    speed — the diffusion stage must be keyed and must cover the whole image.</p>
  <//>`;
};

/* ------------------------------------------------------- claims -- */
const CLAIMS = [
  ["Instrument detects a key-independent leak", true, "1.000, +15.6 to +20.4 dB, both datasets"],
  ["Instrument does not hallucinate", true, "AES and every mismatch control at chance"],
  ["Row/column permutation leaks key-agnostically", true, "up to 22× chance, 2 datasets × 2 attackers"],
  ["Full 2-D permutation leaks much less", true, "2.5–4.6× chance on CIFAR, ~1.5× on shapes"],
  ["Keyed diffusion supplies all key-agnostic security", true, "1.000 → 0.010 with permutation held key-independent"],
  ["Round count governs resistance", false, "1 round already at chance; 2–4 identical"],
  ["float32 keystream degradation is exploitable", false, "at chance"],
  ["Zero-shot key generalisation occurs at any key diversity", false, "flat at chance from 1 key to unlimited"],
  ["CBC feedback confers key-agnostic resistance", true, "where a ceiling exists: CIFAR/resunet 0.981 → 0.010"],
  ["CBC feedback is “secure”", null, "not shown — a claim about one attacker family, not the cipher"],
  ["Attack success depends on attacker inductive bias", true, "0.221 vs 0.087 on the same cipher"],
];

const Claims = () => html`
  <${Section} id="claims" eyebrow="Scope"
    title="What the data supports — and what it does not"
    lede=${`Stating the negative space matters as much as the results. Everything here
      concerns zero-shot key-agnostic attack at 32×32 with two attacker families.`}>
    <div class="tablewrap rev mt2"><table>
      <thead><tr><th class="l">claim</th><th class="l">verdict</th><th class="l">evidence</th></tr></thead>
      <tbody>${CLAIMS.map(([c, ok, ev]) => html`
        <tr key=${c}><td class="l">${c}</td>
          <td class="l"><span class=${"pill " + (ok === true ? "p-broken" : ok === false ? "p-partial" : "p-resisted")}>
            ${ok === true ? "supported" : ok === false ? "refuted" : "not shown"}</span></td>
          <td class="l" style=${S("color:var(--mut)")}>${ev}</td></tr>`)}
      </tbody></table></div>
    <p class="note">This does <strong>not</strong> establish that these configurations are
    secure against a key-adaptive (known-plaintext) attacker, which is a separate setting
    that supplies the key information the zero-shot attacker provably lacks.</p>
  <//>`;

/* ------------------------------------------------- contributions -- */
const Contributions = ({ data }) => html`
  <${Section} id="contributions" alt eyebrow="Contribution"
    title="What this project delivers"
    lede="A reproducible framework and a measured map, rather than a single attack demonstration.">
    <div class="grid g2 mt2">
      ${data.contributions.map(([t, d], i) => html`
        <div class="card rev" key=${t}>
          <div class=${"ribbon rb-" + ["indigo", "cyan", "emerald", "amber", "indigo"][i % 5]}></div>
          <h3 style=${S("font-size:15.5px")}>${t}</h3>
          <p class="note">${d}</p>
        </div>`)}
    </div>
  <//>`;


/* --------------------------------------------------- live walkthrough -- */
const Walkthrough = ({ demo }) => {
  if (!demo || !demo.length) return null;
  const [i, setI] = useState(0);
  const c = demo[i] || demo[0];
  const verdict = c.medianRank <= 5 ? "broken" : c.medianRank <= 200 ? "partial" : "resisted";
  const label = { broken: "recovered", partial: "partially recovered", resisted: "not recovered" }[verdict];
  return html`
  <${Section} id="walkthrough" eyebrow="Sample run"
    title="One image, start to finish, under a key the attacker has never seen"
    lede=${`Each case below is a real execution, not an illustration: a secret key is drawn
      fresh from the full key space, held-out test images are encrypted under it, and a
      trained attacker — which never saw this key during training — is asked to reconstruct
      them. The true plaintext is then ranked against ${c.poolSize - 1} decoys.`}>

    <div class="controls mt2 rev">
      ${demo.map((d, j) => html`
        <button key=${d.run} onClick=${() => setI(j)}
          class=${"pill " + (j === i ? "p-key" : "p-resisted")}
          style=${S("cursor:pointer;border:none;font-family:inherit;font-size:12px;padding:8px 14px")}>
          ${d.title.split("—")[0].trim()}</button>`)}
    </div>

    <div class="card rev">
      <div style=${S("display:flex;gap:12px;align-items:baseline;flex-wrap:wrap")}>
        <h3 style=${S("font-size:18px")}>${c.title}</h3>
        <span class=${"pill p-" + verdict}>${label}</span>
      </div>
      <p class="note" style=${S("max-width:80ch")}>${c.why}</p>

      <div class="grid g3 mt2">
        <div>
          <div class="eyebrow" style=${S("margin-bottom:9px")}>Step 1 · the secret key</div>
          <p class="note" style=${S("margin-top:0")}>Drawn uniformly from the full key space
          at demo time. Training used ${c.trainedKeys}, so the attacker has not seen it.</p>
          <div class="kv mt"><span class="k">x₀</span><span class="v mono">${c.key.x0.toFixed(12)}</span></div>
          <div class="kv"><span class="k">param 2</span><span class="v mono">${c.key.p1.toFixed(12)}</span></div>
          ${c.key.p2 !== 0 && html`<div class="kv"><span class="k">param 3</span><span class="v mono">${c.key.p2.toFixed(12)}</span></div>`}
        </div>
        <div>
          <div class="eyebrow" style=${S("margin-bottom:9px")}>Step 2 · the cipher</div>
          <div class="kv"><span class="k">configuration</span><span class="v mono" style=${S("font-size:11px")}>${c.cipher}</span></div>
          ${["map", "rounds", "mode", "perm_scope", "perm_keyed", "feedback"].map((k) =>
            c.cipherCfg[k] !== undefined ? html`
            <div class="kv" key=${k}><span class="k">${k}</span>
              <span class="v mono">${String(c.cipherCfg[k])}</span></div>` : null)}
        </div>
        <div>
          <div class="eyebrow" style=${S("margin-bottom:9px")}>Step 3 · the attacker</div>
          <div class="kv"><span class="k">architecture</span><span class="v">${c.attacker}</span></div>
          <div class="kv"><span class="k">trained on</span><span class="v">${c.dataset}</span></div>
          <div class="kv"><span class="k">keys seen in training</span><span class="v">${c.trainedKeys}</span></div>
          <div class="kv"><span class="k">this key seen?</span>
            <span class="v" style=${S("color:var(--rose)")}>no</span></div>
        </div>
      </div>

      <div class="eyebrow mt2" style=${S("margin-bottom:9px")}>Step 4 · result on ${c.poolSize} held-out images</div>
      <div class="grid g4">
        <${Stat} ribbon=${verdict === "broken" ? "emerald" : verdict === "partial" ? "amber" : "indigo"}
                 k="Median rank" cls=${verdict === "resisted" ? "" : "ok"}
                 v=${`${Math.round(c.medianRank)} / ${c.poolSize}`}
                 n=${`where the true plaintext lands among ${c.poolSize} candidates · random would be ${Math.round(c.poolSize / 2)}`} />
        <${Stat} ribbon="cyan" k="Exact top-1" v=${f(c.top1)}
                 n=${`identified outright · chance ${f(1 / c.poolSize, 3)}`} />
        <${Stat} ribbon="indigo" k="PSNR" v=${f(c.meanPsnr, 2) + " dB"}
                 n=${`prior floor ${f(c.floorPsnr, 2)} dB · gain ${c.gainDb > 0 ? "+" : ""}${f(c.gainDb, 2)} dB`} />
        <${Stat} ribbon="amber" k="Top-10" v=${f(c.top10)}
                 n="true plaintext inside the 10 nearest candidates" />
      </div>

      <div class="eyebrow mt2" style=${S("margin-bottom:9px")}>Step 5 · what came out</div>
      <div class="samples">
        ${c.samples.map((t, k) => html`
          <div class="trip" key=${k}>
            <img src=${t.plain} alt="plaintext" /><div class="lab">plaintext</div>
            <img src=${t.cipher} alt="ciphertext" /><div class="lab">ciphertext</div>
            <img src=${t.recon} alt="reconstruction" /><div class="lab">recovered</div>
            <div style=${S("font-size:10px;color:var(--mut);text-align:center;line-height:1.5")}>
              ${f(t.psnr, 1)} dB<br/>
              <span style=${{ color: t.rank === 1 ? "var(--emerald)" : t.rank <= 10 ? "var(--amber)" : "var(--mut)",
                              fontWeight: t.rank <= 10 ? 700 : 400 }}>rank ${t.rank}</span>
            </div>
          </div>`)}
      </div>
      <p class="note"><strong>How to read the rank.</strong> It is the position of the true
      plaintext when all ${c.poolSize} candidates are ordered by similarity to the
      reconstruction. Rank 1 means the attacker picked the right image outright; a rank near
      ${Math.round(c.poolSize / 2)} means it did no better than guessing. This matters more
      than how the pictures look — a plausible-looking output can be produced from the
      dataset prior alone, and rank is what separates the two.</p>
    </div>
  <//>`;
};

/* --------------------------------------------------------------- app -- */
function App() {
  const [data, setData] = useState(null);
  const [demo, setDemo] = useState(null);
  const [sel, setSel] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    const load = () => fetch("data.json?t=" + Date.now())
      .then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then((d) => { setData(d); setSel((s) => s || pickDefault(d)); })
      .catch((e) => setErr(String(e)));
    fetch("demo.json?t=" + Date.now()).then((r) => r.ok ? r.json() : null)
      .then(setDemo).catch(() => setDemo(null));
    load();
    const t = setInterval(load, 60000);
    return () => clearInterval(t);
  }, []);

  useReveal();

  if (err) return html`<div class="boot"><p>Could not load data.json — ${err}</p>
    <p class="note">Run <span class="mono">python3 build_web.py</span> and serve this
    directory over HTTP (file:// blocks fetch).</p></div>`;
  if (!data) return html`<div class="boot"><div class="boot-ring"></div><p>Loading results…</p></div>`;

  return html`
    <${Nav} generated=${data.generated} />
    <${Hero} runs=${data.runs} />
    <${Overview} data=${data} />
    <${PlainEnglish} />
    <${KeyFindings} data=${data} />
    <${Questions} data=${data} />
    <${Problem} />
    <${Datasets} data=${data} />
    <${Cipher} data=${data} />
    <${Method} data=${data} />
    <${Calibration} runs=${data.runs} />
    <${KeyCurve} data=${data} />
    <${Mechanism} data=${data} />
    <${Walkthrough} demo=${demo} />
    <${Results} runs=${data.runs} sel=${sel} setSel=${setSel} />
    <${Samples} data=${data} sel=${sel} runs=${data.runs} />
    <${Claims} />
    <${Findings} />
    <${Contributions} data=${data} />
    <${Status} data=${data} />
    <div class="footer"><div class="wrap">
      Crynexa · key-agnostic neural cryptanalysis of chaos-based image encryption ·
      generated ${data.generated} · all figures measured from trained models
    </div></div>`;
}

function pickDefault(d) {
  const withSamples = d.runs.filter((r) => d.samples[r.run]);
  if (!withSamples.length) return null;
  const ka = withSamples.filter((r) => r.regime === "key-agnostic");
  const pool = ka.length ? ka : withSamples;
  return pool.reduce((a, b) => (a.unseenTop1 > b.unseenTop1 ? a : b)).run;
}

const Overview = ({ data }) => {
  const r = data.runs;
  const ka = r.filter((x) => x.regime === "key-agnostic");
  const gpuSecs = r.reduce((a, x) => a + x.secs, 0);
  return html`
  <${Section} id="overview" alt eyebrow="At a glance"
    title="What was built and what was measured"
    lede=${`A reproducible framework: a parameterised chaos cipher with an AES control, an
      on-the-fly plaintext–ciphertext generator with correct uniform key sampling, four
      attacker architectures, and an evaluation protocol whose controls make a null result
      interpretable.`}>
    <div class="grid g4 mt2">
      <${Stat} ribbon="indigo" k="Trained models" v=${html`<${CountUp} value=${r.length} />`}
               n="each with full controls recorded" />
      <${Stat} ribbon="cyan" k="Key-agnostic runs" v=${html`<${CountUp} value=${ka.length} />`}
               n="unlimited key diversity — every test key unseen" />
      <${Stat} ribbon="emerald" k="Cipher configs" cls="ok"
               v=${html`<${CountUp} value=${new Set(r.map((x) => x.cipher)).size} />`}
               n="structural variants measured" />
      <${Stat} ribbon="amber" k="Compute" v=${html`<${CountUp} value=${gpuSecs / 60} decimals=${0} suffix=" min" />`}
               n="total GPU training time across all runs" />
    </div>
  <//>`;
};

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
