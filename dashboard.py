#!/usr/bin/env python3
"""Build a self-contained HTML results dashboard.

    python dashboard.py            # -> reports/dashboard.html
    python dashboard.py --open     # and open it

Everything (data, charts, sample images) is embedded, so the file works from
disk with no server and no network.
"""
import argparse
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"


# ---------------------------------------------------------------- data ----
def png_b64(arr, scale=3):
    """uint8 HWC array -> base64 PNG data URI, nearest-neighbour upscaled."""
    from PIL import Image
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    im = Image.fromarray(arr.astype(np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_name(name):
    """E2_<variant>__<regime>__<attacker>_<dataset>  (older names degrade ok)"""
    suite = name.split("_")[0]
    regime = ("same-key" if "__samekey" in name else
              "key-agnostic" if "__keyagnostic" in name else "-")
    attacker = "-"
    for a in ("resunet", "linearmixer", "pixelmixer", "cnn"):
        if a in name:
            attacker = a
    dataset = "-"
    for d in ("shapes", "cifar10", "noise", "stl10"):
        if name.endswith("_" + d) or f"_{d}_" in name:
            dataset = d
    variant = name
    for cut in ("__samekey", "__keyagnostic"):
        if cut in variant:
            variant = variant.split(cut)[0]
    variant = variant[len(suite) + 1:] if variant.startswith(suite + "_") else variant
    return suite, variant, regime, attacker, dataset


def collect():
    rows, samples = [], {}
    for d in sorted(RUNS.iterdir()):
        f = d / "result.json"
        if not f.is_file():
            continue
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        c, s, u, fl, mm = r["config"], r["seen"], r["unseen"], r["floor"], r["mismatch_control"]
        suite, variant, regime, attacker, dataset = parse_name(c["name"])
        rows.append({
            "run": c["name"], "suite": suite, "variant": variant,
            "regime": regime, "attacker": attacker or c["model"],
            "dataset": dataset if dataset != "-" else c["dataset"],
            "cipher": r["cipher_tag"],
            "n_keys": c["n_keys"] if c["n_keys"] else "unlimited",
            "steps": c["steps"], "params_m": round(r["n_params"] / 1e6, 1),
            "seen_top1": s["top1_pool100"], "unseen_top1": u["top1_pool100"],
            "mismatch_top1": mm["top1_pool100"], "chance": s["chance_pool100"],
            "seen_top1_1k": s.get("top1_pool1000"), "unseen_top1_1k": u.get("top1_pool1000"),
            "seen_psnr": s["psnr"], "unseen_psnr": u["psnr"], "floor_psnr": fl["psnr"],
            "gain_db": u["psnr"] - fl["psnr"],
            "seen_ssim": s["ssim"], "unseen_ssim": u["ssim"],
            "kgg": r["kgg_psnr"],
            "secs": round(r["train_seconds"]),
            "history": [{"step": h["step"],
                         "seen": h.get("seen", {}).get("top1_pool100"),
                         "unseen": h["unseen"].get("top1_pool100"),
                         "unseen_psnr": h["unseen"]["psnr"]} for h in r["history"]],
        })
        npz = d / "samples.npz"
        if npz.is_file():
            try:
                z = np.load(npz)
                samples[c["name"]] = [
                    {"plain": png_b64(z["plain"][i]), "cipher": png_b64(z["cipher"][i]),
                     "recon": png_b64(z["recon"][i])} for i in range(min(6, len(z["plain"])))]
            except Exception:
                pass
    return rows, samples


# ---------------------------------------------------------------- html ----
CSS = """
:root{
  --bg:#f6f7f9; --panel:#fff; --ink:#12151a; --muted:#5b6472; --line:#e2e6ec;
  --accent:#2b6cb0; --good:#1a7f5a; --bad:#b4342b; --warn:#8a6d1f;
  --chip:#eef1f6; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0e1116; --panel:#161b22; --ink:#e6edf3; --muted:#8b949e; --line:#283039;
  --accent:#6aa6e8; --good:#3fb984; --bad:#f0685e; --warn:#d2a63c; --chip:#1d242d;
}}
:root[data-theme="dark"]{
  --bg:#0e1116; --panel:#161b22; --ink:#e6edf3; --muted:#8b949e; --line:#283039;
  --accent:#6aa6e8; --good:#3fb984; --bad:#f0685e; --warn:#d2a63c; --chip:#1d242d;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
     margin:0;padding:28px 22px 64px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1360px;margin:0 auto}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:15px;margin:34px 0 12px;letter-spacing:-.01em;
   text-transform:uppercase;font-weight:650;color:var(--muted)}
.sub{color:var(--muted);margin:0 0 22px;font-size:13px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px 18px}
.grid{display:grid;gap:13px}
.cards{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.card .k{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:25px;font-weight:640;margin-top:5px;font-variant-numeric:tabular-nums}
.card .n{color:var(--muted);font-size:12px;margin-top:3px}
.ok{color:var(--good)} .no{color:var(--bad)} .wn{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:12.6px;font-variant-numeric:tabular-nums}
th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child,th.l,td.l{text-align:left}
th{cursor:pointer;user-select:none;color:var(--muted);font-weight:600;font-size:11.5px;
   text-transform:uppercase;letter-spacing:.04em;position:sticky;top:0;background:var(--panel)}
th:hover{color:var(--ink)}
tbody tr:hover{background:var(--chip)}
.mono{font-family:var(--mono);font-size:11.5px}
.chip{display:inline-block;background:var(--chip);border-radius:5px;padding:1px 7px;
      font-size:11px;color:var(--muted)}
.bar{height:7px;border-radius:4px;background:var(--chip);overflow:hidden;min-width:70px;display:inline-block;vertical-align:middle}
.bar>i{display:block;height:100%;background:var(--accent)}
.controls{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:12px;align-items:center}
select,input{background:var(--panel);color:var(--ink);border:1px solid var(--line);
             border-radius:7px;padding:6px 9px;font:inherit;font-size:12.5px}
.scroll{overflow-x:auto;max-height:660px;overflow-y:auto}
.samples{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:11px}
.trip{text-align:center}
.trip img{display:block;width:100%;image-rendering:pixelated;border-radius:5px;border:1px solid var(--line)}
.trip .lab{font-size:10px;color:var(--muted);margin:3px 0 6px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}
.note{color:var(--muted);font-size:12.5px;margin-top:9px;line-height:1.6}
.empty{color:var(--muted);padding:26px;text-align:center}
"""

JS = r"""
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const f=(v,d=3)=>v==null?'-':(typeof v==='number'?v.toFixed(d):v);
let sortKey='unseen_top1', sortDir=-1;

function chanceClass(r){
  if(r.unseen_top1 >= 5*r.chance) return 'ok';
  if(r.unseen_top1 >= 2*r.chance) return 'wn';
  return 'no';
}
function filtered(){
  const q=$('#q').value.toLowerCase(), ds=$('#ds').value, at=$('#at').value, rg=$('#rg').value;
  return ROWS.filter(r=>(!q||r.run.toLowerCase().includes(q))
    &&(ds==='all'||r.dataset===ds)&&(at==='all'||r.attacker===at)&&(rg==='all'||r.regime===rg))
    .sort((a,b)=>{const x=a[sortKey],y=b[sortKey];
      if(x==null)return 1; if(y==null)return -1;
      return (typeof x==='number'?x-y:String(x).localeCompare(String(y)))*sortDir;});
}
const COLS=[['variant','config',1],['regime','keys',1],['attacker','attacker',1],
  ['dataset','data',1],['n_keys','#keys',1],['seen_top1','seen top1',0],
  ['unseen_top1','unseen top1',0],['mismatch_top1','mismatch',0],
  ['unseen_psnr','unseen psnr',0],['floor_psnr','floor',0],['gain_db','gain dB',0],
  ['kgg','KGG',0],['secs','sec',0]];

function render(){
  const rows=filtered();
  $('#count').textContent=rows.length+' runs';
  $('#thead').innerHTML='<tr>'+COLS.map(([k,l,isL])=>
    `<th class="${isL?'l':''}" data-k="${k}">${l}${sortKey===k?(sortDir>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>';
  $('#tbody').innerHTML = rows.length? rows.map(r=>{
    const w=Math.min(100,r.unseen_top1*100);
    return `<tr data-run="${r.run}">
      <td class="l mono" title="${r.cipher}">${r.variant}</td>
      <td class="l"><span class="chip">${r.regime}</span></td>
      <td class="l">${r.attacker}</td><td class="l">${r.dataset}</td>
      <td>${r.n_keys}</td>
      <td>${f(r.seen_top1)}</td>
      <td class="${chanceClass(r)}"><span class="bar"><i style="width:${w}%"></i></span> ${f(r.unseen_top1)}</td>
      <td>${f(r.mismatch_top1)}</td>
      <td>${f(r.unseen_psnr,2)}</td><td>${f(r.floor_psnr,2)}</td>
      <td class="${r.gain_db>1?'ok':(r.gain_db<-0.5?'no':'')}">${f(r.gain_db,2)}</td>
      <td>${f(r.kgg,3)}</td><td>${r.secs}</td></tr>`;}).join('')
    : '<tr><td colspan="13" class="empty">no runs match</td></tr>';
  $$('#thead th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; sortDir = (k===sortKey)? -sortDir : -1; sortKey=k; render();});
  $$('#tbody tr').forEach(tr=>tr.onclick=()=>showSamples(tr.dataset.run));
  drawBars(rows);
}

function drawBars(rows){
  const el=$('#bars'); const R=rows.filter(r=>r.regime==='key-agnostic').slice(0,26);
  if(!R.length){el.innerHTML='<div class="empty">no key-agnostic runs yet</div>';return;}
  const H=R.length*23+34, W=760, L=250, max=Math.max(0.06,...R.map(r=>r.unseen_top1));
  const x=v=>L+(v/max)*(W-L-70), ch=R[0].chance;
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">
   ${R.map((r,i)=>{const y=i*23+16;const cls=chanceClass(r);
     const col=cls==='ok'?'var(--good)':cls==='wn'?'var(--warn)':'var(--accent)';
     return `<text x="${L-8}" y="${y+11}" text-anchor="end" font-size="11" fill="var(--muted)"
              font-family="var(--mono)">${r.variant.slice(0,34)}</text>
       <rect x="${L}" y="${y+2}" width="${Math.max(1,x(r.unseen_top1)-L)}" height="15" rx="3" fill="${col}"/>
       <text x="${x(r.unseen_top1)+7}" y="${y+13}" font-size="11" fill="var(--muted)">${r.unseen_top1.toFixed(3)}</text>`;}).join('')}
   <line x1="${x(ch)}" y1="10" x2="${x(ch)}" y2="${H-16}" stroke="var(--bad)" stroke-width="1.4" stroke-dasharray="4 3"/>
   <text x="${x(ch)+5}" y="${H-4}" font-size="10.5" fill="var(--bad)">chance ${ch}</text></svg>`;
}

function showSamples(run){
  const s=SAMPLES[run]; const el=$('#samples');
  $('#sampleTitle').textContent = run;
  if(!s){el.innerHTML='<div class="empty">no saved samples for this run</div>';return;}
  el.innerHTML=s.map(t=>`<div class="trip">
     <img src="${t.plain}"><div class="lab">plaintext</div>
     <img src="${t.cipher}"><div class="lab">ciphertext</div>
     <img src="${t.recon}"><div class="lab">reconstruction</div></div>`).join('');
}

function init(){
  const uniq=(k)=>[...new Set(ROWS.map(r=>r[k]))].sort();
  for(const [sel,k] of [['#ds','dataset'],['#at','attacker'],['#rg','regime']])
    $(sel).innerHTML='<option value="all">all</option>'+uniq(k).map(v=>`<option>${v}</option>`).join('');
  ['#q','#ds','#at','#rg'].forEach(s=>$(s).addEventListener('input',render));
  render();
  const best=filtered().find(r=>SAMPLES[r.run]);
  if(best) showSamples(best.run);
}
"""


def cards(rows):
    def find(pred):
        m = [r for r in rows if pred(r)]
        return max(m, key=lambda r: r["unseen_top1"]) if m else None

    pos = find(lambda r: "posctrl" in r["run"] or "calpos" in r["run"])
    aes = find(lambda r: "aes" in r["run"] or r["cipher"] == "aes")
    ka = [r for r in rows if r["regime"] == "key-agnostic"]
    broken = [r for r in ka if r["unseen_top1"] >= 5 * r["chance"]]

    out = []
    if pos:
        ok = pos["unseen_top1"] > 0.5
        out.append(("Positive control", f"{pos['unseen_top1']:.3f}",
                    "key-independent cipher, unseen keys — must break",
                    "ok" if ok else "no"))
    if aes:
        ok = aes["unseen_top1"] < 5 * aes["chance"]
        out.append(("AES-CTR control", f"{aes['unseen_top1']:.3f}",
                    "must stay at chance — else the pipeline leaks",
                    "ok" if ok else "no"))
    out.append(("Key-agnostic runs", str(len(ka)),
                "cipher configs attacked with unlimited key diversity", ""))
    out.append(("Configs broken", f"{len(broken)}/{len(ka)}" if ka else "0",
                "unseen-key retrieval ≥ 5× chance",
                "no" if broken else "ok"))
    return out


def build(rows, samples):
    REPORTS.mkdir(exist_ok=True)
    cs = cards(rows)
    card_html = "".join(
        f'<div class="panel card"><div class="k">{k}</div>'
        f'<div class="v {cl}">{v}</div><div class="n">{n}</div></div>'
        for k, v, n, cl in cs)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<title>Crynexa Results</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Crynexa — key-agnostic neural cryptanalysis</h1>
<p class="sub">{len(rows)} runs · generated {ts} · all numbers measured, none synthetic</p>

<h2>Calibration</h2>
<div class="grid cards">{card_html}</div>
<p class="note">The instrument is only trustworthy when the positive control breaks
<em>and</em> the AES control does not. A null result on a keyed cipher means nothing
unless both hold, and unless that cipher's own same-key ceiling is above chance.</p>

<h2>Key-agnostic attack success by cipher configuration</h2>
<div class="panel" id="bars"></div>
<div class="legend">
  <span><i style="background:var(--good)"></i>≥5× chance — broken</span>
  <span><i style="background:var(--warn)"></i>2–5× chance — partial</span>
  <span><i style="background:var(--accent)"></i>&lt;2× chance — resisted</span>
</div>

<h2>All runs</h2>
<div class="controls">
  <input id="q" placeholder="filter by name…" style="min-width:220px">
  <select id="ds"></select><select id="at"></select><select id="rg"></select>
  <span class="chip" id="count"></span>
  <span class="note" style="margin:0">click a row to load its samples</span>
</div>
<div class="panel scroll"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>

<h2>Reconstructions — <span id="sampleTitle" class="mono" style="text-transform:none"></span></h2>
<div class="panel"><div class="samples" id="samples"></div></div>
<p class="note">Top row plaintext, middle ciphertext, bottom the attacker's reconstruction.
Judge these against the retrieval column, not by eye: a plausible-looking image can be
produced from the dataset prior alone, which is what the mismatch control measures.</p>
</div>
<script>
const ROWS={json.dumps(rows)};
const SAMPLES={json.dumps(samples)};
{JS}
init();
</script>"""
    p = REPORTS / "dashboard.html"
    p.write_text(html)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    rows, samples = collect()
    p = build(rows, samples)
    size = p.stat().st_size / 1e6
    print(f"{len(rows)} runs, {len(samples)} with samples -> {p} ({size:.1f} MB)")
    if a.open:
        import webbrowser
        webbrowser.open(f"file://{p}")


if __name__ == "__main__":
    main()
