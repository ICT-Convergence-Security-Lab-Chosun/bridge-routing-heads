"""visualization.py — Unified visualization entry point for multilingual_hop.

Select one or more visualizations via CLI flags:

  --step5_stage2_5_attention   Interactive HTML attention heatmap
  --step4_bhs                  Interactive HTML layer×head BHS score heatmap  (+ PNG/PDF)
  --step5_stage2_ablation      Interactive HTML mean-ablation bar chart          (+ PNG/PDF)
  --step2_5_pearson            Pearson correlation table for KL divergence       (+ PNG/PDF)
  --all                        Run all of the above

Per-visualization hyperparameters live in CONFIGS["models"] lists.
Each model generates its own output file: e.g. step4_bhs_llama31_70.html

Usage
-----
    python script/visualization.py --step5_stage2_5_attention
    python script/visualization.py --step4_bhs --step5_stage2_ablation
    python script/visualization.py --step2_5_pearson
    python script/visualization.py --all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from scipy.stats import pearsonr as _pearsonr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "output" / "visualization"

# ============================================================
# HYPERPARAMETERS  (edit models list and other settings here)
# ============================================================

CONFIGS: dict[str, dict] = {

    # ── Step 4 : Bridge Head Score layer×head heatmap ────────────────────────────────────
    "step4_bhs": {
        "models":           ["llama31_70", "qwen25_72"],
        "input_root":       PROJECT_ROOT / "output" / "step4_filtering_Bridge_head_Score",
        # which formula's bridge_scores_*.jsonl to use
        "formula":          "fh+th-sh",          # fh+th-sh | th-fh-sh
        "langs":            ["en", "ko", "zh", "ja", "es"],
        # metrics available as tabs in the heatmap sidebar
        "metrics":          ["BHS", "z_FH", "z_TH", "z_SH"],
        "default_metric":   "BHS",
        # colour scale:  viridis | plasma | coolwarm
        "default_colormap": "viridis",
        # top-K heads to highlight with a marker dot
        "top_k":            512,
        # canvas cell size in pixels
        "cell_w":           14,
        "cell_h":           9,
    },

    # ── Step 5 Stage 2 : Mean Ablation delta scores bar chart ────────────────────
    "step5_stage2_ablation": {
        "models":           ["llama31_70", "qwen25_72"],
        "input_root":       PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes",
        "langs":            ["ko", "zh", "ja", "es"],
        # which delta metrics to show as sub-bars
        "metrics":          ["delta_XL", "delta_Mono", "delta_FH", "delta_SH"],
    },

    # ── Step 5 Stage 2.5 : Question-Scoped Attention Heatmap ──────────────────
    "step5_stage2_5_attention": {
        "models":           ["llama31_70"],
        "input_root":       PROJECT_ROOT / "output" / "step5_BridgeHead_Ablation_Patchscopes",
        "stage_dir":        "stage2.5_attention",
        "langs":            ["ko", "zh", "ja", "es"],

        # highlight colour: yellow | green | cyan | orange | pink
        "heatmap_color":    "yellow",
        # 1 = single column,  2 = two columns
        "layout_cols":      2,
        # per_record : each card normalised independently
        # per_head   : all 50 cards share the same max weight
        "normalize_mode":   "per_record",
        # how many records to show per (lang, head)  (<=50)
        "max_records":      50,
    },

    # ── Step 2.5 : Activation Patching KL Pearson Correlation ─────────────────
    "step2_5_pearson": {
        "models":           ["llama31_70", "qwen25_72"],
        "input_root":       PROJECT_ROOT / "output" / "step2_5_activation_patching",
        "modules":          ["mlp", "attn"],
        # Files actually present on disk: average_{pair}.json
        "lang_pairs":       ["en_ko", "en_zh", "ko_zh"],
        # 6 pairwise comparisons requested; pairs without data show N/A
        "corr_pairs": [
            ("en_ko", "en_zh"),
            ("en_ko", "ko_en"),
            ("en_ko", "ko_zh"),
            ("en_zh", "ko_en"),
            ("en_zh", "ko_zh"),
            ("ko_en", "ko_zh"),
        ],
        # which patch position's KL to use
        "position":         "answer_colon",
    },

}

# ============================================================
# Shared utilities
# ============================================================

_HIGHLIGHT_COLORS: dict[str, tuple[int, int, int]] = {
    "yellow": (255, 230,   0),
    "green":  (  0, 220,  80),
    "cyan":   (  0, 210, 230),
    "orange": (255, 140,   0),
    "pink":   (255, 100, 180),
}


def _save_html(html: str, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / f"{name}.html"
    p.write_text(html, encoding="utf-8")
    print(f"  Saved -> {p}  ({p.stat().st_size // 1024} KB)")


def _inject(template: str, replacements: dict[str, str]) -> str:
    """Replace @@KEY@@ tokens in template."""
    for key, val in replacements.items():
        template = template.replace(f"@@{key}@@", val)
    return template


def _embed_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</script>", r"<\/script>")


def _mpl_pub_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.linewidth":    0.8,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def _save_fig(fig, slug: str) -> None:
    """Save figure as PNG and PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"{slug}.{ext}"
        fig.savefig(str(p), format=ext, bbox_inches="tight")
        print(f"  Saved -> {p}")

_ATTENTION_HTML = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8">\n<title>Attention Heatmap · @@MODEL_SHORT@@</title>\n<style>\n*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\nbody{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#eceff4;color:#222;display:flex;height:100vh;overflow:hidden}\n#sidebar{width:230px;min-width:180px;background:#1a1d2e;color:#dce0ef;display:flex;flex-direction:column;padding:18px 14px 14px;gap:14px;overflow-y:auto;flex-shrink:0;border-right:2px solid #0d0f1a}\n.sb-title{font-size:12px;font-weight:700;letter-spacing:1px;color:#82b1ff;text-transform:uppercase;border-bottom:1px solid #2e3150;padding-bottom:10px}\n.ctrl-lbl{font-size:10px;color:#7986cb;text-transform:uppercase;letter-spacing:.6px;font-weight:600}\n.ctrl-blk{display:flex;flex-direction:column;gap:6px}\n.lang-grp{display:flex;flex-wrap:wrap;gap:5px}\n.lang-btn{padding:4px 12px;border-radius:4px;border:1px solid #3a3f6e;background:#252945;color:#b0bec5;cursor:pointer;font-size:12px;font-weight:700;transition:background .12s}\n.lang-btn:hover{background:#323870}.lang-btn.active{background:#1565c0;border-color:#1e88e5;color:#fff}\n#head-select{width:100%;min-height:150px;background:#252945;color:#cfd8dc;border:1px solid #3a3f6e;border-radius:5px;padding:5px 8px;font-size:12.5px;font-family:"SF Mono","Fira Code",monospace;line-height:1.8;outline:none}\n#head-select option:checked{background:#1565c0}\n.legend-bar{height:10px;border-radius:3px;background:linear-gradient(to right,rgba(@@CSTR@@,.05),rgba(@@CSTR@@,.45),rgba(@@CSTR@@,1));border:1px solid #2e3150}\n.legend-lbls{display:flex;justify-content:space-between;font-size:10px;color:#546e7a;margin-top:3px}\n.info-blk{font-size:10.5px;color:#546e7a;line-height:1.85}.info-blk b{color:#78909c}\n#main{flex:1;display:flex;flex-direction:column;overflow:hidden;background:#eceff4}\n#topbar{background:#fff;border-bottom:1px solid #d0d7e0;padding:9px 20px;display:flex;align-items:center;gap:14px;flex-shrink:0;min-height:42px}\n#tb-title{font-size:14px;font-weight:700;color:#263238;white-space:nowrap}\n#tb-sub{font-size:12px;color:#607d8b;font-family:"SF Mono","Fira Code",monospace}\n#tb-count{margin-left:auto;font-size:11px;color:#90a4ae;white-space:nowrap}\n#scroll-area{flex:1;overflow-y:auto;padding:16px 20px 24px}\n#prompt-grid{display:grid;grid-template-columns:repeat(@@COLS@@,1fr);gap:10px}\n.prompt-card{background:#fff;border:1px solid #dde2ea;border-radius:8px;padding:9px 13px;font-size:14px;line-height:2.0;word-break:break-word}\n.card-meta{font-size:10px;color:#b0bec5;font-family:"SF Mono","Fira Code",monospace;margin-bottom:4px}\n.tok{border-radius:3px;padding:0 1px;display:inline;white-space:pre-wrap}\n.plain{display:inline;white-space:pre-wrap;color:#546e7a}\n.ph{grid-column:1/-1;text-align:center;color:#b0bec5;padding:80px 20px;font-size:15px}\n</style></head><body>\n<div id="sidebar">\n  <div class="sb-title">Attention Heatmap</div>\n  <div class="ctrl-blk"><span class="ctrl-lbl">Language</span><div class="lang-grp" id="lang-btns"></div></div>\n  <div class="ctrl-blk" style="flex:1;display:flex;flex-direction:column"><span class="ctrl-lbl">Layer / Head</span><select id="head-select" size="12"></select></div>\n  <div class="ctrl-blk"><span class="ctrl-lbl">Intensity</span><div class="legend-bar"></div><div class="legend-lbls"><span>Low</span><span>High</span></div></div>\n  <div class="info-blk"><b>Model:</b> @@MODEL_SHORT@@<br><b>Stage:</b> @@STAGE_DIR@@<br><b>Norm:</b> @@NORMALIZE_MODE@@<br><b>Color:</b> @@HEATMAP_COLOR@@</div>\n</div>\n<div id="main">\n  <div id="topbar"><span id="tb-title">Prompt Attention View</span><span id="tb-sub"></span><span id="tb-count"></span></div>\n  <div id="scroll-area"><div id="prompt-grid"><div class="ph">← Select a language and head to begin.</div></div></div>\n</div>\n<script>\n"use strict";\nconst DATA      = @@DATA_JSON@@;\nconst COLOR     = [@@COLOR_R@@, @@COLOR_G@@, @@COLOR_B@@];\nconst NORM_MODE = "@@NORMALIZE_MODE@@";\nlet curLang=null, curHead=null;\nfunction initLangs(){\n  const c=document.getElementById("lang-btns"), langs=Object.keys(DATA);\n  langs.forEach(lg=>{\n    const b=document.createElement("button");\n    b.className="lang-btn"; b.textContent=lg.toUpperCase(); b.dataset.lang=lg;\n    b.addEventListener("click",()=>setLang(lg)); c.appendChild(b);\n  });\n  if(langs.length)setLang(langs[0]);\n}\nfunction setLang(lang){\n  if(!DATA[lang])return; curLang=lang;\n  document.querySelectorAll(".lang-btn").forEach(b=>b.classList.toggle("active",b.dataset.lang===lang));\n  buildHeadList(lang);\n}\nfunction buildHeadList(lang){\n  const sel=document.getElementById("head-select"); sel.innerHTML="";\n  Object.entries(DATA[lang]).forEach(([key,info])=>{\n    const o=document.createElement("option"); o.value=key;\n    o.textContent="L"+String(info.layer).padStart(2,"0")+"  H"+String(info.head).padStart(2,"0");\n    sel.appendChild(o);\n  });\n  sel.onchange=()=>setHead(sel.value);\n  if(sel.options.length){sel.selectedIndex=0;setHead(sel.options[0].value);}\n}\nfunction setHead(hk){if(!curLang||!DATA[curLang][hk])return;curHead=hk;renderGrid(curLang,hk);}\nfunction applyNorm(recs){\n  if(NORM_MODE==="per_head"){\n    let mx=1e-15; recs.forEach(r=>r.tokens.forEach(t=>{if(t.w>mx)mx=t.w;}));\n    return recs.map(r=>({...r,tokens:r.tokens.map(t=>({...t,n:t.w/mx}))}));\n  }\n  return recs.map(r=>{\n    let mx=1e-15; r.tokens.forEach(t=>{if(t.w>mx)mx=t.w;});\n    return{...r,tokens:r.tokens.map(t=>({...t,n:t.w/mx}))};\n  });\n}\nfunction esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}\nfunction tokSpan(text,n){\n  const [r,g,b]=COLOR, a=(Math.sqrt(n)*.78+(n>.015?.07:0)).toFixed(3);\n  return"<span class=\\"tok\\" style=\\"background:rgba("+r+","+g+","+b+","+a+")\\" title=\\""+(text.trim()||"(space)")+" w="+n.toFixed(4)+"\\">"+esc(text)+"</span>";\n}\nfunction buildCard(rec,idx){\n  const pre=rec.prefix?"<span class=\\"plain\\">"+esc(rec.prefix)+"</span>":"";\n  const suf=rec.suffix?"<span class=\\"plain\\">"+esc(rec.suffix)+"</span>":"";\n  const body=rec.tokens.map(t=>tokSpan(t.t,t.n)).join("");\n  const d=document.createElement("div"); d.className="prompt-card";\n  d.innerHTML="<div class=\\"card-meta\\">#"+(idx+1)+"&nbsp;&nbsp;"+esc(rec.hop_id||"")+"</div><div>"+pre+body+suf+"</div>";\n  return d;\n}\nfunction renderGrid(lang,hk){\n  const info=DATA[lang][hk], recs=applyNorm(info.records);\n  document.getElementById("tb-sub").textContent=lang.toUpperCase()+"  \\u00b7  Layer "+info.layer+"  \\u00b7  Head "+info.head;\n  document.getElementById("tb-count").textContent=recs.length+" records";\n  const grid=document.getElementById("prompt-grid"), frag=document.createDocumentFragment();\n  recs.forEach((rec,i)=>frag.appendChild(buildCard(rec,i)));\n  grid.innerHTML=""; grid.appendChild(frag);\n}\ninitLangs();\n</script></body></html>'

_BHS_HTML = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8">\n<title>BHS Heatmap · @@MODEL_SHORT@@</title>\n<style>\n*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\nbody{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#1a1d2e;color:#dce0ef;display:flex;height:100vh;overflow:hidden}\n#sidebar{width:200px;min-width:160px;background:#12142a;display:flex;flex-direction:column;padding:16px 12px;gap:12px;overflow-y:auto;flex-shrink:0;border-right:2px solid #0d0f1a}\n.sb-title{font-size:12px;font-weight:700;letter-spacing:1px;color:#82b1ff;text-transform:uppercase;border-bottom:1px solid #2e3150;padding-bottom:9px}\n.ctrl-lbl{font-size:10px;color:#7986cb;text-transform:uppercase;letter-spacing:.6px;font-weight:600;margin-bottom:4px;display:block}\n.btn-grp{display:flex;flex-wrap:wrap;gap:4px}\n.sel-btn{padding:3px 9px;border-radius:4px;border:1px solid #3a3f6e;background:#1e2240;color:#b0bec5;cursor:pointer;font-size:11.5px;transition:background .1s}\n.sel-btn:hover{background:#2e3460}.sel-btn.active{background:#1565c0;border-color:#1e88e5;color:#fff}\n.info-blk{font-size:10px;color:#546e7a;line-height:1.85;margin-top:auto}.info-blk b{color:#78909c}\n#main{flex:1;display:flex;flex-direction:column;overflow:hidden;background:#f5f6fa}\n#topbar{background:#fff;border-bottom:1px solid #d0d7e0;padding:8px 18px;display:flex;align-items:center;gap:12px;flex-shrink:0}\n#tb-title{font-size:14px;font-weight:700;color:#263238}\n#tb-sub{font-size:12px;color:#607d8b;font-family:monospace}\n#tb-count{margin-left:auto;font-size:11px;color:#90a4ae}\n#canvas-outer{flex:1;overflow:auto;padding:20px 24px;display:flex;flex-direction:column;align-items:flex-start;gap:0}\n#axis-x-wrap{margin-left:42px}\n#axis-x-canvas{display:block}\n#canvas-row{display:flex;align-items:flex-start}\n#axis-y-canvas{flex-shrink:0;display:block}\n#heatmap{display:block;cursor:crosshair;image-rendering:pixelated;image-rendering:crisp-edges}\n#cb-wrap{display:flex;flex-direction:column;gap:3px;margin-top:6px;margin-left:42px}\n#cb-canvas{display:block;border-radius:3px}\n#cb-lbls{display:flex;justify-content:space-between;font-size:10px;color:#607d8b}\n#tooltip{position:fixed;background:rgba(10,12,30,.92);color:#e0e8ff;padding:7px 11px;border-radius:5px;font-size:12px;font-family:monospace;pointer-events:none;display:none;z-index:9999;white-space:pre;line-height:1.6}\n</style></head><body>\n<div id="sidebar">\n  <div class="sb-title">BHS Heatmap</div>\n  <div><span class="ctrl-lbl">Language</span><div class="btn-grp" id="lang-btns"></div></div>\n  <div><span class="ctrl-lbl">Metric</span><div class="btn-grp" id="metric-btns"></div></div>\n  <div><span class="ctrl-lbl">Colormap</span><div class="btn-grp" id="cmap-btns"></div></div>\n  <div style="font-size:11px;color:#546e7a;line-height:1.7">&#9733; Top-<b id="topk-lbl" style="color:#82b1ff"></b> heads<br><span style="font-size:10px">hover for values</span></div>\n  <div class="info-blk"><b>Model:</b> @@MODEL_SHORT@@<br><b>Formula:</b> @@FORMULA@@</div>\n</div>\n<div id="main">\n  <div id="topbar">\n    <span id="tb-title">Bridge Head Score Heatmap</span>\n    <span id="tb-sub"></span><span id="tb-count"></span>\n  </div>\n  <div id="canvas-outer">\n    <div id="axis-x-wrap"><canvas id="axis-x-canvas" height="22"></canvas></div>\n    <div id="canvas-row">\n      <canvas id="axis-y-canvas" width="42"></canvas>\n      <canvas id="heatmap"></canvas>\n    </div>\n    <div id="cb-wrap">\n      <canvas id="cb-canvas" height="14"></canvas>\n      <div id="cb-lbls"><span id="cb-min"></span><span id="cb-max"></span></div>\n    </div>\n  </div>\n</div>\n<div id="tooltip"></div>\n<script>\n"use strict";\nconst DATA    = @@DATA_JSON@@;\nconst CELL_W  = @@CELL_W@@;\nconst CELL_H  = @@CELL_H@@;\nconst METRICS = @@METRICS_JSON@@;\nconst TOP_K   = @@TOP_K@@;\nconst CMAPS   = {\n  viridis:  [[.267,.005,.329],[.190,.407,.574],[.208,.719,.474],[.993,.906,.144]],\n  plasma:   [[.050,.030,.528],[.490,.012,.657],[.903,.393,.228],[.940,.975,.131]],\n  coolwarm: [[.231,.299,.752],[.865,.865,.865],[.706,.016,.150]]\n};\nfunction mapColor(t,cmap){\n  const s=CMAPS[cmap], n=s.length-1;\n  const i=Math.min(Math.floor(t*n),n-1), f=t*n-i;\n  return s[i].map((v,j)=>Math.round((v+f*(s[i+1][j]-v))*255));\n}\nfunction clamp01(v,mn,mx){return mx===mn?0.5:(v-mn)/(mx-mn);}\n\nlet curLang=Object.keys(DATA)[0], curMetric="@@DEFAULT_METRIC@@", curCmap="@@DEFAULT_CMAP@@";\n\nfunction mkBtns(ids,cid,onSet,getCur){\n  const c=document.getElementById(cid);\n  ids.forEach(id=>{\n    const b=document.createElement("button");\n    b.className="sel-btn"+(getCur()===id?" active":"");\n    b.textContent=id; b.dataset.id=id;\n    b.addEventListener("click",()=>{\n      onSet(id);\n      document.querySelectorAll("#"+cid+" .sel-btn").forEach(x=>x.classList.toggle("active",x.dataset.id===id));\n      render();\n    });\n    c.appendChild(b);\n  });\n}\nfunction initButtons(){\n  mkBtns(Object.keys(DATA),"lang-btns",v=>{curLang=v;},()=>curLang);\n  mkBtns(METRICS,"metric-btns",v=>{curMetric=v;},()=>curMetric);\n  mkBtns(Object.keys(CMAPS),"cmap-btns",v=>{curCmap=v;},()=>curCmap);\n  document.getElementById("topk-lbl").textContent=TOP_K;\n}\n\nfunction drawAxisX(n_heads){\n  const cv=document.getElementById("axis-x-canvas");\n  cv.width=n_heads*CELL_W;\n  const ctx=cv.getContext("2d");\n  ctx.clearRect(0,0,cv.width,cv.height);\n  ctx.fillStyle="#607d8b"; ctx.font="9px monospace"; ctx.textAlign="center";\n  const step=Math.max(4,Math.ceil(32/CELL_W)*4);\n  for(let h=0;h<n_heads;h+=step) ctx.fillText(h,h*CELL_W+CELL_W/2,12);\n  ctx.textAlign="center"; ctx.fillText("head",n_heads*CELL_W/2,22);\n}\nfunction drawAxisY(n_layers){\n  const cv=document.getElementById("axis-y-canvas");\n  cv.height=n_layers*CELL_H;\n  const ctx=cv.getContext("2d");\n  ctx.clearRect(0,0,cv.width,cv.height);\n  ctx.fillStyle="#607d8b"; ctx.font="9px monospace"; ctx.textAlign="right";\n  const step=Math.max(4,Math.ceil(24/CELL_H)*4);\n  for(let l=0;l<n_layers;l+=step) ctx.fillText(l,38,l*CELL_H+CELL_H/2+3);\n}\nfunction drawColorbar(mn,mx,n_heads){\n  const cv=document.getElementById("cb-canvas");\n  cv.width=n_heads*CELL_W;\n  const ctx=cv.getContext("2d"), W=cv.width;\n  for(let x=0;x<W;x++){\n    const [r,g,b]=mapColor(x/W,curCmap);\n    ctx.fillStyle="rgb("+r+","+g+","+b+")"; ctx.fillRect(x,0,1,14);\n  }\n  document.getElementById("cb-min").textContent=mn.toFixed(4);\n  document.getElementById("cb-max").textContent=mx.toFixed(4);\n}\nfunction render(){\n  const info=DATA[curLang]; if(!info)return;\n  const {n_layers,n_heads,top_k}=info, vals=info[curMetric];\n  const mn=Math.min.apply(null,vals), mx=Math.max.apply(null,vals);\n  drawAxisX(n_heads); drawAxisY(n_layers); drawColorbar(mn,mx,n_heads);\n  const cv=document.getElementById("heatmap");\n  cv.width=n_heads*CELL_W; cv.height=n_layers*CELL_H;\n  const ctx=cv.getContext("2d");\n  const topSet=new Set(top_k.map(function(p){return p[0]*n_heads+p[1];}));\n  for(let layer=0;layer<n_layers;layer++){\n    for(let head=0;head<n_heads;head++){\n      const idx=layer*n_heads+head;\n      const t=clamp01(vals[idx],mn,mx);\n      const [r,g,b]=mapColor(t,curCmap);\n      ctx.fillStyle="rgb("+r+","+g+","+b+")";\n      ctx.fillRect(head*CELL_W,layer*CELL_H,CELL_W,CELL_H);\n      if(topSet.has(idx)){\n        ctx.fillStyle="rgba(255,255,255,0.72)";\n        ctx.beginPath();\n        ctx.arc(head*CELL_W+CELL_W/2,layer*CELL_H+CELL_H/2,Math.min(CELL_W,CELL_H)/5,0,Math.PI*2);\n        ctx.fill();\n      }\n    }\n  }\n  document.getElementById("tb-sub").textContent=curLang.toUpperCase()+"  \\u00b7  "+curMetric+"  \\u00b7  "+curCmap;\n  document.getElementById("tb-count").textContent=n_layers+"L \\u00d7 "+n_heads+"H";\n}\nconst tooltip=document.getElementById("tooltip");\ndocument.getElementById("heatmap").addEventListener("mousemove",function(e){\n  const info=DATA[curLang]; if(!info)return;\n  const {n_heads,n_layers,top_k}=info;\n  const rect=e.currentTarget.getBoundingClientRect();\n  const head=Math.floor((e.clientX-rect.left)/CELL_W);\n  const layer=Math.floor((e.clientY-rect.top)/CELL_H);\n  if(head<0||head>=n_heads||layer<0||layer>=n_layers)return;\n  const idx=layer*n_heads+head;\n  const isTop=top_k.some(function(p){return p[0]===layer&&p[1]===head;});\n  const lines=METRICS.map(function(m){return"  "+m.padEnd(8)+": "+info[m][idx].toFixed(4);});\n  tooltip.style.display="block";\n  tooltip.style.left=(e.clientX+14)+"px";\n  tooltip.style.top=(e.clientY-8)+"px";\n  tooltip.textContent="L"+layer+" H"+head+(isTop?" \\u2605":"")+"\\n"+lines.join("\\n");\n});\ndocument.getElementById("heatmap").addEventListener("mouseleave",function(){tooltip.style.display="none";});\ninitButtons(); render();\n</script></body></html>'

_ABLATION_HTML = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8">\n<title>Ablation Scores · @@MODEL_SHORT@@</title>\n<style>\n*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\nbody{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f5f6fa;color:#222;display:flex;height:100vh;overflow:hidden}\n#sidebar{width:210px;min-width:160px;background:#1a1d2e;color:#dce0ef;display:flex;flex-direction:column;padding:16px 12px;gap:12px;overflow-y:auto;flex-shrink:0;border-right:2px solid #0d0f1a}\n.sb-title{font-size:12px;font-weight:700;letter-spacing:1px;color:#82b1ff;text-transform:uppercase;border-bottom:1px solid #2e3150;padding-bottom:9px}\n.ctrl-lbl{font-size:10px;color:#7986cb;text-transform:uppercase;letter-spacing:.6px;font-weight:600;margin-bottom:4px;display:block}\n.btn-grp{display:flex;flex-wrap:wrap;gap:5px}\n.lang-btn{padding:4px 12px;border-radius:4px;border:1px solid #3a3f6e;background:#252945;color:#b0bec5;cursor:pointer;font-size:12px;font-weight:700;transition:background .1s}\n.lang-btn:hover{background:#323870}.lang-btn.active{background:#1565c0;border-color:#1e88e5;color:#fff}\n.legend{display:flex;flex-direction:column;gap:5px;font-size:11px;color:#78909c}\n.legend-row{display:flex;align-items:center;gap:7px}\n.dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}\n.ctrl-check{display:flex;align-items:center;gap:6px;font-size:12px;color:#b0bec5;cursor:pointer}\n.ctrl-check input{cursor:pointer;accent-color:#1e88e5}\n.stats-blk{font-size:11px;color:#546e7a;line-height:1.9;margin-top:auto}.stats-blk b{color:#90a4ae}\n#main{flex:1;display:flex;flex-direction:column;overflow:hidden}\n#topbar{background:#fff;border-bottom:1px solid #d0d7e0;padding:8px 18px;display:flex;align-items:center;gap:14px;flex-shrink:0}\n#tb-title{font-size:14px;font-weight:700;color:#263238}\n#tb-sub{font-size:12px;color:#607d8b;font-family:monospace}\n#tb-count{margin-left:auto;font-size:11px;color:#90a4ae}\n#scroll-area{flex:1;overflow-y:auto;padding:12px 20px 20px}\n#metric-legend{display:flex;flex-wrap:wrap;gap:14px;background:#fff;border:1px solid #e0e4ec;border-radius:6px;padding:7px 14px;margin-bottom:10px}\n.ml-item{display:flex;align-items:center;gap:5px;font-size:11.5px;color:#607d8b}\n.ml-dot{width:11px;height:11px;border-radius:2px;flex-shrink:0}\n.head-row{display:grid;grid-template-columns:90px 50px 1fr 100px;align-items:center;gap:8px;padding:5px 10px;background:#fff;border:1px solid #e0e4ec;border-left:3px solid transparent;border-radius:5px;margin-bottom:3px;min-height:38px}\n.head-row.passed{border-left-color:#43a047}\n.head-row.failed{border-left-color:#e0e0e0;opacity:.72}\n.head-id{font-size:12px;font-family:monospace;color:#455a64;white-space:nowrap}\n.badge{font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;text-align:center;letter-spacing:.3px}\n.badge.pass{background:#e8f5e9;color:#2e7d32}.badge.fail{background:#f5f5f5;color:#9e9e9e}\n.bars{display:flex;flex-direction:column;gap:2px}\n.bar-row{display:flex;align-items:center;gap:5px;height:10px}\n.bar-lbl{font-size:9px;color:#90a4ae;width:48px;text-align:right;flex-shrink:0}\n.bipolar{position:relative;flex:1;height:8px;background:#f0f0f0;border-radius:2px;overflow:hidden}\n.zero-line{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#bdbdbd}\n.bar-fill{position:absolute;top:0;bottom:0;border-radius:2px}\n.vals{font-size:10px;font-family:monospace;color:#607d8b;line-height:1.7;white-space:nowrap}\n</style></head><body>\n<div id="sidebar">\n  <div class="sb-title">Ablation Scores</div>\n  <div><span class="ctrl-lbl">Language</span><div class="btn-grp" id="lang-btns"></div></div>\n  <div>\n    <span class="ctrl-lbl">Filter</span>\n    <label class="ctrl-check"><input type="checkbox" id="only-passed"> Show passed only</label>\n  </div>\n  <div class="legend">\n    <div class="legend-row"><div class="dot" style="background:#43a047"></div><span>Passed (&Delta;XL&gt;0)</span></div>\n    <div class="legend-row"><div class="dot" style="background:#e0e0e0"></div><span>Failed</span></div>\n  </div>\n  <div style="font-size:11px;color:#546e7a;line-height:1.7">Pass criterion:<br><span style="font-family:monospace;color:#82b1ff">delta_XL &gt; 0</span></div>\n  <div class="stats-blk" id="stats-blk"></div>\n</div>\n<div id="main">\n  <div id="topbar">\n    <span id="tb-title">Mean Ablation Score</span>\n    <span id="tb-sub"></span><span id="tb-count"></span>\n  </div>\n  <div id="scroll-area">\n    <div id="metric-legend">\n      <div class="ml-item"><div class="ml-dot" style="background:#43a047"></div>&Delta; XL (pos)</div>\n      <div class="ml-item"><div class="ml-dot" style="background:#ef5350"></div>&Delta; XL (neg)</div>\n      <div class="ml-item"><div class="ml-dot" style="background:#42a5f5"></div>&Delta; Mono</div>\n      <div class="ml-item"><div class="ml-dot" style="background:#ab47bc"></div>&Delta; FH</div>\n      <div class="ml-item"><div class="ml-dot" style="background:#ff7043"></div>&Delta; SH</div>\n    </div>\n    <div id="head-list"></div>\n  </div>\n</div>\n<script>\n"use strict";\nconst DATA    = @@DATA_JSON@@;\nconst METRICS = @@METRICS_JSON@@;\nlet curLang=Object.keys(DATA)[0];\n\nfunction initLangs(){\n  const c=document.getElementById("lang-btns");\n  Object.keys(DATA).forEach(function(lg){\n    const b=document.createElement("button");\n    b.className="lang-btn"+(lg===curLang?" active":"");\n    b.textContent=lg.toUpperCase(); b.dataset.lang=lg;\n    b.addEventListener("click",function(){\n      curLang=lg;\n      document.querySelectorAll(".lang-btn").forEach(function(x){x.classList.toggle("active",x.dataset.lang===lg);});\n      render();\n    });\n    c.appendChild(b);\n  });\n}\ndocument.getElementById("only-passed").addEventListener("change",render);\n\nfunction biBar(delta,maxAbs,posColor,negColor){\n  const pct=Math.min(Math.abs(delta)/(maxAbs||1)*50,50).toFixed(1);\n  if(delta>=0){\n    return"<div class=\\"bar-fill\\" style=\\"background:"+posColor+";left:50%;width:"+pct+"%\\"></div>";\n  } else {\n    return"<div class=\\"bar-fill\\" style=\\"background:"+negColor+";right:50%;width:"+pct+"%\\"></div>";\n  }\n}\n\nfunction sign(v){return v>0?"+":"";}\n\nfunction buildRow(h,mXL,mMono,mFH,mSH){\n  const dXL=h.delta_XL||0, dMono=h.delta_Mono||0, dFH=h.delta_FH||0, dSH=h.delta_SH||0;\n  const passed=h.passed;\n  const row=document.createElement("div");\n  row.className="head-row "+(passed?"passed":"failed");\n  row.innerHTML=\n    "<span class=\\"head-id\\">L"+String(h.layer).padStart(2,"0")+" H"+String(h.head).padStart(2,"0")+"</span>"+\n    "<span class=\\"badge "+(passed?"pass":"fail")+"\\">"+(passed?"PASS":"FAIL")+"</span>"+\n    "<div class=\\"bars\\">"+\n      "<div class=\\"bar-row\\"><span class=\\"bar-lbl\\">\\u0394 XL</span><div class=\\"bipolar\\"><div class=\\"zero-line\\"></div>"+biBar(dXL,mXL,"#43a047","#ef5350")+"</div></div>"+\n      "<div class=\\"bar-row\\"><span class=\\"bar-lbl\\">\\u0394 Mono</span><div class=\\"bipolar\\"><div class=\\"zero-line\\"></div>"+biBar(dMono,mMono,"#42a5f5","#42a5f5")+"</div></div>"+\n      "<div class=\\"bar-row\\"><span class=\\"bar-lbl\\">\\u0394 FH</span><div class=\\"bipolar\\"><div class=\\"zero-line\\"></div>"+biBar(dFH,mFH,"#ab47bc","#ab47bc")+"</div></div>"+\n      "<div class=\\"bar-row\\"><span class=\\"bar-lbl\\">\\u0394 SH</span><div class=\\"bipolar\\"><div class=\\"zero-line\\"></div>"+biBar(dSH,mSH,"#ff7043","#ff7043")+"</div></div>"+\n    "</div>"+\n    "<div class=\\"vals\\">XL:&nbsp; "+sign(dXL)+dXL.toFixed(4)+"<br>Mo:&nbsp; "+sign(dMono)+dMono.toFixed(4)+"<br>FH:&nbsp; "+sign(dFH)+dFH.toFixed(4)+"<br>SH:&nbsp; "+sign(dSH)+dSH.toFixed(4)+"</div>";\n  return row;\n}\n\nfunction render(){\n  const info=DATA[curLang]; if(!info)return;\n  const onlyPassed=document.getElementById("only-passed").checked;\n  const heads=onlyPassed?info.heads.filter(function(h){return h.passed;}):info.heads;\n  function mabs(fn){return Math.max.apply(null,[1e-9].concat(heads.map(fn)));}\n  const mXL  =mabs(function(h){return Math.abs(h.delta_XL||0);});\n  const mMono=mabs(function(h){return Math.abs(h.delta_Mono||0);});\n  const mFH  =mabs(function(h){return Math.abs(h.delta_FH||0);});\n  const mSH  =mabs(function(h){return Math.abs(h.delta_SH||0);});\n  const list=document.getElementById("head-list"), frag=document.createDocumentFragment();\n  heads.forEach(function(h){frag.appendChild(buildRow(h,mXL,mMono,mFH,mSH));});\n  list.innerHTML=""; list.appendChild(frag);\n  document.getElementById("tb-sub").textContent=curLang.toUpperCase();\n  document.getElementById("tb-count").textContent=heads.length+" heads";\n  document.getElementById("stats-blk").innerHTML=\n    "<b>Total:</b> "+info.n_candidates+"<br><b>Passed:</b> "+info.n_passed+"<br>"+\n    "<b>Rate:</b> "+(info.n_passed/info.n_candidates*100).toFixed(1)+"%";\n}\n\ninitLangs(); render();\n</script></body></html>'

_PEARSON_HTML = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8"><title>KL Pearson · @@MODEL_SHORT@@</title>\n<style>*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f5f6fa;color:#222;display:flex;height:100vh;overflow:hidden}\n#sidebar{width:190px;min-width:160px;background:#1a1d2e;color:#dce0ef;display:flex;flex-direction:column;padding:16px 12px;gap:12px;overflow-y:auto;flex-shrink:0;border-right:2px solid #0d0f1a}\n.sb-title{font-size:12px;font-weight:700;letter-spacing:1px;color:#82b1ff;text-transform:uppercase;border-bottom:1px solid #2e3150;padding-bottom:9px}\n.ctrl-lbl{font-size:10px;color:#7986cb;text-transform:uppercase;letter-spacing:.6px;font-weight:600;margin-bottom:4px;display:block}\n.btn-grp{display:flex;flex-wrap:wrap;gap:5px}.mod-btn{padding:4px 12px;border-radius:4px;border:1px solid #3a3f6e;background:#252945;color:#b0bec5;cursor:pointer;font-size:12px;font-weight:700;transition:background .1s}.mod-btn:hover{background:#323870}.mod-btn.active{background:#1565c0;border-color:#1e88e5;color:#fff}\n.info-blk{font-size:10.5px;color:#546e7a;line-height:1.85;margin-top:auto}.info-blk b{color:#78909c}\n#main{flex:1;display:flex;flex-direction:column;overflow:hidden}#topbar{background:#fff;border-bottom:1px solid #d0d7e0;padding:8px 18px;display:flex;align-items:center;gap:14px;flex-shrink:0}\n#tb-title{font-size:14px;font-weight:700;color:#263238}#tb-sub{font-size:12px;color:#607d8b;font-family:monospace;margin-left:8px}\n#content{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:20px}\n.section-card{background:#fff;border:1px solid #e0e4ec;border-radius:8px;padding:16px}\n.section-title{font-size:13px;font-weight:700;color:#263238;margin-bottom:12px}\n.corr-table{border-collapse:collapse;width:100%;font-size:13px}\n.corr-table th{background:#263238;color:#eceff1;padding:8px 14px;text-align:center;font-weight:600;font-size:12px;letter-spacing:.3px;white-space:nowrap}\n.corr-table td{padding:9px 14px;text-align:center;border-bottom:1px solid #eee}\n.corr-table tr:last-child td{border-bottom:none}\n.r-val{font-family:monospace;font-weight:700;font-size:15px}\n.p-val{font-size:11px;color:#607d8b;font-family:monospace;margin-top:2px}\n.sig{font-size:12px;font-weight:700}\n.pair-cell{font-family:monospace;font-size:13px;color:#37474f;text-align:left;padding-left:18px!important}\n.na-cell{color:#bdbdbd;font-style:italic;font-size:12px}\n.kl-canvas{display:block}\n.legend-row{display:flex;gap:18px;margin-top:10px;flex-wrap:wrap}\n.leg-item{display:flex;align-items:center;gap:6px;font-size:11.5px;color:#607d8b}\n.leg-line{width:22px;height:3px;border-radius:2px}\n</style></head><body>\n<div id="sidebar">\n  <div class="sb-title">KL Pearson</div>\n  <div><span class="ctrl-lbl">Module</span><div class="btn-grp" id="mod-btns"></div></div>\n  <div class="info-blk"><b>Model:</b> @@MODEL_SHORT@@<br><b>KL position:</b> @@POSITION@@<br><b>Metric:</b> answer_colon KL</div>\n</div>\n<div id="main">\n  <div id="topbar">\n    <span id="tb-title">Activation Patching — KL Pearson Correlation</span>\n    <span id="tb-sub"></span>\n  </div>\n  <div id="content">\n    <div class="section-card">\n      <div class="section-title">Pearson Correlation Table (answer_colon KL divergence)</div>\n      <div id="table-wrap"></div>\n    </div>\n    <div class="section-card">\n      <div class="section-title">Per-Layer KL Divergence Curves</div>\n      <canvas id="kl-chart" class="kl-canvas"></canvas>\n      <div class="legend-row" id="chart-legend"></div>\n    </div>\n  </div>\n</div>\n<script>\n"use strict";\nvar DATA = @@DATA_JSON@@;\nvar CORR_PAIRS = @@CORR_PAIRS_JSON@@;\nvar PAIR_COLORS = {"en_ko":"#1565c0","en_zh":"#2e7d32","ko_zh":"#b71c1c","ko_en":"#6a1b9a","en_ja":"#e65100","ko_ja":"#37474f"};\nvar curModule = Object.keys(DATA)[0];\nfunction initMods() {\n  var c = document.getElementById("mod-btns");\n  Object.keys(DATA).forEach(function(mod) {\n    var b = document.createElement("button");\n    b.className = "mod-btn" + (mod === curModule ? " active" : "");\n    b.textContent = mod.toUpperCase(); b.dataset.mod = mod;\n    b.addEventListener("click", function() {\n      curModule = mod;\n      document.querySelectorAll(".mod-btn").forEach(function(x) { x.classList.toggle("active", x.dataset.mod === mod); });\n      render();\n    });\n    c.appendChild(b);\n  });\n}\nfunction sigStars(p) {\n  if (p === null || p === undefined) return "";\n  if (p < 0.001) return "***";\n  if (p < 0.01)  return "**";\n  if (p < 0.05)  return "*";\n  return "ns";\n}\nfunction rBg(r) {\n  if (r === null || r === undefined) return "";\n  var v = Math.abs(r);\n  if (v >= 0.9) return "background:rgba(76,175,80,.25)";\n  if (v >= 0.7) return "background:rgba(139,195,74,.18)";\n  if (v >= 0.5) return "background:rgba(255,213,79,.22)";\n  if (v >= 0.3) return "background:rgba(255,152,0,.15)";\n  return "background:rgba(239,83,80,.10)";\n}\nfunction sigColor(sig) {\n  if (sig === "***") return "color:#c62828";\n  if (sig === "**")  return "color:#e53935";\n  if (sig === "*")   return "color:#ef5350";\n  return "color:#90a4ae";\n}\nfunction buildTable(mod) {\n  var corrs = DATA[mod].correlations;\n  var rows = CORR_PAIRS.map(function(pair) {\n    var key = pair[0] + "_vs_" + pair[1];\n    var entry = corrs[key];\n    var a = pair[0].replace("_", "→"), b = pair[1].replace("_", "→");\n    if (!entry) {\n      return `<tr><td class="pair-cell">${a}</td><td class="pair-cell">${b}</td><td colspan="3" class="na-cell">N/A — data not available</td></tr>`;\n    }\n    var r = entry.r, p = entry.p, sig = sigStars(p);\n    var pStr = p < 0.0001 ? "&lt;0.0001" : p.toFixed(4);\n    return `<tr style="${rBg(r)}"><td class="pair-cell">${a}</td><td class="pair-cell">${b}</td><td><span class="r-val">${r.toFixed(4)}</span></td><td><span class="p-val">${pStr}</span></td><td><span class="sig" style="${sigColor(sig)}">${sig}</span></td></tr>`;\n  }).join("");\n  return `<table class="corr-table"><thead><tr><th>Lang Pair A</th><th>Lang Pair B</th><th>Pearson r</th><th>p-value</th><th>Sig.</th></tr></thead><tbody>${rows}</tbody></table>`;\n}\nfunction drawChart(mod) {\n  var cv = document.getElementById("kl-chart");\n  var W = (cv.parentElement.clientWidth - 32) || 700, H = 230;\n  cv.width = W; cv.height = H;\n  var ctx = cv.getContext("2d");\n  ctx.clearRect(0, 0, W, H);\n  var curves = DATA[mod].curves;\n  var pairs = Object.keys(curves);\n  if (!pairs.length) return;\n  var n = Math.max.apply(null, pairs.map(function(p) { return curves[p].length; }));\n  var allV = [].concat.apply([], pairs.map(function(p) { return curves[p]; }));\n  var mn = Math.min.apply(null, allV), mx = Math.max.apply(null, allV);\n  var pad = {l:58, r:16, t:12, b:28};\n  var gW = W - pad.l - pad.r, gH = H - pad.t - pad.b;\n  ctx.strokeStyle = "#e8eaf0"; ctx.lineWidth = 0.6;\n  for (var i = 0; i <= 4; i++) {\n    var y0 = pad.t + gH * (1 - i / 4);\n    ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(pad.l + gW, y0); ctx.stroke();\n    ctx.fillStyle = "#90a4ae"; ctx.font = "9px monospace"; ctx.textAlign = "right";\n    ctx.fillText((mn + i / 4 * (mx - mn)).toExponential(2), pad.l - 3, y0 + 3);\n  }\n  ctx.fillStyle = "#90a4ae"; ctx.font = "9px monospace"; ctx.textAlign = "center";\n  var xStep = Math.max(4, Math.round(n / 10));\n  for (var j = 0; j < n; j += xStep) ctx.fillText(j, pad.l + j / (n - 1 || 1) * gW, H - 4);\n  ctx.fillText("Layer", pad.l + gW / 2, H);\n  pairs.forEach(function(pair) {\n    var vals = curves[pair], color = PAIR_COLORS[pair] || "#546e7a";\n    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round";\n    ctx.beginPath();\n    vals.forEach(function(v, k) {\n      var x0 = pad.l + k / (vals.length - 1 || 1) * gW;\n      var y1 = pad.t + gH * (1 - (v - mn) / (mx - mn || 1));\n      if (k === 0) ctx.moveTo(x0, y1); else ctx.lineTo(x0, y1);\n    });\n    ctx.stroke();\n  });\n  var legDiv = document.getElementById("chart-legend");\n  legDiv.innerHTML = "";\n  pairs.forEach(function(pair) {\n    var color = PAIR_COLORS[pair] || "#546e7a";\n    var d = document.createElement("div"); d.className = "leg-item";\n    d.innerHTML = `<div class="leg-line" style="background:${color}"></div>${pair.replace("_","→")}`;\n    legDiv.appendChild(d);\n  });\n}\nfunction render() {\n  document.getElementById("tb-sub").textContent = curModule.toUpperCase();\n  document.getElementById("table-wrap").innerHTML = buildTable(curModule);\n  drawChart(curModule);\n}\ninitMods(); render();\nwindow.addEventListener("resize", function() { if (curModule) drawChart(curModule); });\n</script></body></html>'

# ============================================================
# HTML templates (embedded below)
# ============================================================
# (HTML template variables are defined above this block)


# ============================================================
# Viz 1 — step5_stage2_5_attention
# ============================================================

def _clean_token(tok: str) -> str:
    tok = tok.replace("\u2581", " ")
    tok = re.sub(r"<0x[0-9A-Fa-f]+>", "\u00b7", tok)
    return tok


def _load_attention_data(cfg: dict) -> dict:
    data: dict = {}
    base = cfg["input_root"] / cfg["model_short"] / cfg["stage_dir"]
    for lang in cfg["langs"]:
        path = base / lang / "question_attn_weights.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        lang_data: dict = {}
        for head_key, head_info in raw["heads"].items():
            layer = head_info["layer"]
            head  = head_info["head"]
            records: list[dict] = []
            for rec in head_info["records"][: cfg["max_records"]]:
                raw_q  = rec.get("raw_question_text", "")
                body_q = rec.get("question_text", raw_q)
                idx = raw_q.find(body_q)
                prefix = raw_q[:idx]                if idx >= 0 else ""
                suffix = raw_q[idx + len(body_q):]  if idx >= 0 else ""
                tokens = [
                    {"t": _clean_token(t["token"]), "w": float(t["attn_weight"])}
                    for t in rec["question_tokens"]
                ]
                records.append({
                    "prefix": prefix,
                    "suffix": suffix,
                    "tokens": tokens,
                    "hop_id": rec.get("hop_id", ""),
                })
            lang_data[head_key] = {"layer": layer, "head": head, "records": records}
        lang_data = dict(sorted(lang_data.items(), key=lambda x: (x[1]["layer"], x[1]["head"])))
        data[lang] = lang_data
        n_recs = len(next(iter(lang_data.values()))["records"]) if lang_data else 0
        print(f"  [{lang}] {len(lang_data)} heads, {n_recs} records each")
    return data


def _run_attention(cfg: dict) -> None:
    for model in cfg["models"]:
        print(f"[step5_stage2_5_attention] model={model}")
        m_cfg = {**cfg, "model_short": model}
        data = _load_attention_data(m_cfg)
        if not data:
            print("  No data found — skipping.")
            continue
        color_rgb = _HIGHLIGHT_COLORS.get(cfg["heatmap_color"], _HIGHLIGHT_COLORS["yellow"])
        cstr = f"{color_rgb[0]},{color_rgb[1]},{color_rgb[2]}"
        html = _inject(_ATTENTION_HTML, {
            "DATA_JSON":      _embed_json(data),
            "COLOR_R":        str(color_rgb[0]),
            "COLOR_G":        str(color_rgb[1]),
            "COLOR_B":        str(color_rgb[2]),
            "CSTR":           cstr,
            "MODEL_SHORT":    model,
            "STAGE_DIR":      cfg["stage_dir"],
            "NORMALIZE_MODE": cfg["normalize_mode"],
            "HEATMAP_COLOR":  cfg["heatmap_color"],
            "COLS":           str(max(1, min(2, cfg["layout_cols"]))),
        })
        _save_html(html, f"step5_stage2_5_attention_{model}")


# ============================================================
# Viz 2 — step4_bhs
# ============================================================

def _load_bhs_data(cfg: dict) -> dict:
    data: dict = {}
    root = cfg["input_root"] / cfg["model_short"] / cfg["formula"]
    for lang in cfg["langs"]:
        path = root / lang / f"bridge_scores_{lang}.jsonl"
        if not path.exists():
            continue
        records: dict[tuple[int, int], dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                records[(r["layer"], r["head"])] = r
        if not records:
            continue
        n_layers = max(k[0] for k in records) + 1
        n_heads  = max(k[1] for k in records) + 1
        lang_data: dict = {"n_layers": n_layers, "n_heads": n_heads}
        for metric in cfg["metrics"]:
            flat = [0.0] * (n_layers * n_heads)
            for (layer, head), rec in records.items():
                flat[layer * n_heads + head] = round(float(rec.get(metric, 0.0)), 4)
            lang_data[metric] = flat
        top = sorted(records.values(), key=lambda r: r.get("BHS", 0.0), reverse=True)
        lang_data["top_k"] = [[r["layer"], r["head"]] for r in top[: cfg["top_k"]]]
        data[lang] = lang_data
        print(f"  [{lang}] {n_layers}L x {n_heads}H  ({len(records)} heads loaded)")
    return data


def _save_bhs_static(data: dict, cfg: dict, slug: str) -> None:
    """Save BHS heatmap as publication-quality PNG + PDF."""
    if not _HAS_MPL or not _HAS_NUMPY:
        print("  [static] matplotlib/numpy not available — skipping PNG/PDF")
        return
    langs = [l for l in cfg["langs"] if l in data]
    if not langs:
        return
    _mpl_pub_style()
    n = len(langs)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 3.2 * n), 5))
    if n == 1:
        axes = [axes]
    metric = cfg.get("default_metric", "BHS")
    for ax, lang in zip(axes, langs):
        info = data[lang]
        mat = np.array(info[metric], dtype=float).reshape(info["n_layers"], info["n_heads"])
        im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="upper",
                       interpolation="nearest")
        ax.set_title(lang.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Head", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Layer", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    model   = cfg["model_short"]
    formula = cfg.get("formula", "")
    fig.suptitle(f"Bridge Head Score ({metric}) — {model}  [{formula}]",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, slug)
    plt.close(fig)


def _run_bhs(cfg: dict) -> None:
    for model in cfg["models"]:
        print(f"[step4_bhs] model={model}")
        m_cfg = {**cfg, "model_short": model}
        data = _load_bhs_data(m_cfg)
        if not data:
            print("  No data found — skipping.")
            continue
        slug = f"step4_bhs_{model}"
        html = _inject(_BHS_HTML, {
            "DATA_JSON":      _embed_json(data),
            "CELL_W":         str(cfg["cell_w"]),
            "CELL_H":         str(cfg["cell_h"]),
            "METRICS_JSON":   json.dumps(cfg["metrics"]),
            "TOP_K":          str(cfg["top_k"]),
            "DEFAULT_METRIC": cfg["default_metric"],
            "DEFAULT_CMAP":   cfg["default_colormap"],
            "MODEL_SHORT":    model,
            "FORMULA":        cfg["formula"],
        })
        _save_html(html, slug)
        _save_bhs_static(data, m_cfg, slug)


# ============================================================
# Viz 3 — step5_stage2_ablation
# ============================================================

def _load_ablation_data(cfg: dict) -> dict:
    data: dict = {}
    root = cfg["input_root"] / cfg["model_short"] / "stage2_ablation"
    for lang in cfg["langs"]:
        path = root / lang / "ablation_scores.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        heads = list(d["heads"].values())
        for h in heads:
            h["passed"] = bool(h.get("delta_XL", 0.0) > 0.0)
        heads.sort(key=lambda h: h.get("delta_XL", 0.0), reverse=True)
        data[lang] = {
            "heads":        heads,
            "n_candidates": d.get("n_candidates", len(heads)),
            "n_passed":     sum(1 for h in heads if h["passed"]),
        }
        print(f"  [{lang}] {len(heads)} candidates, {data[lang]['n_passed']} passed")
    return data


def _save_ablation_static(data: dict, cfg: dict, slug: str) -> None:
    """Save ablation bar chart as publication-quality PNG + PDF."""
    if not _HAS_MPL:
        print("  [static] matplotlib not available — skipping PNG/PDF")
        return
    from matplotlib.patches import Patch
    langs = [l for l in cfg["langs"] if l in data]
    if not langs:
        return
    _mpl_pub_style()
    n = len(langs)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 3.5 * n), 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, lang in zip(axes, langs):
        info = data[lang]
        heads = info["heads"]
        y = [h.get("delta_XL", 0.0) for h in heads]
        colors = ["#43a047" if h["passed"] else "#ef5350" for h in heads]
        ax.bar(range(len(heads)), y, color=colors, width=1.0, linewidth=0, zorder=2)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=3)
        ax.set_title(lang.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Head rank", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("\u0394XL", fontsize=9)
        ax.text(0.97, 0.96, f"pass={info['n_passed']}/{info['n_candidates']}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#546e7a")
    handles = [Patch(facecolor="#43a047", label="Pass (\u0394XL>0)"),
               Patch(facecolor="#ef5350", label="Fail (\u0394XL\u22640)")]
    fig.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)
    fig.suptitle(f"Mean Ablation \u0394XL — {cfg['model_short']}", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, slug)
    plt.close(fig)


def _run_ablation(cfg: dict) -> None:
    for model in cfg["models"]:
        print(f"[step5_stage2_ablation] model={model}")
        m_cfg = {**cfg, "model_short": model}
        data = _load_ablation_data(m_cfg)
        if not data:
            print("  No data found — skipping.")
            continue
        slug = f"step5_stage2_ablation_{model}"
        html = _inject(_ABLATION_HTML, {
            "DATA_JSON":    _embed_json(data),
            "METRICS_JSON": json.dumps(cfg["metrics"]),
            "MODEL_SHORT":  model,
        })
        _save_html(html, slug)
        _save_ablation_static(data, m_cfg, slug)


# ============================================================
# Viz 4 — step2_5_pearson
# ============================================================

def _load_pearson_data(cfg: dict, model_short: str) -> dict:
    """Returns {module: {lang_pair: [kl_per_layer]}}."""
    result: dict = {}
    base = cfg["input_root"] / model_short
    position = cfg.get("position", "answer_colon")
    for module in cfg["modules"]:
        mod_dir = base / module
        if not mod_dir.exists():
            continue
        curves: dict = {}
        for pair in cfg["lang_pairs"]:
            path = mod_dir / f"average_{pair}.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text(encoding="utf-8"))
            kl = d.get("per_layer_kl", {})
            vals = kl.get(position, []) if isinstance(kl, dict) else kl
            if vals:
                curves[pair] = [float(v) for v in vals]
        result[module] = curves
        print(f"  [{model_short}/{module}] {len(curves)} pairs: {list(curves.keys())}")
    return result


def _compute_correlations(curves: dict, corr_pairs: list) -> dict:
    """Compute Pearson r for each requested corr pair."""
    out: dict = {}
    if not _HAS_SCIPY:
        print("  scipy not available — Pearson correlation skipped")
        for pa, pb in corr_pairs:
            out[f"{pa}_vs_{pb}"] = None
        return out
    for pa, pb in corr_pairs:
        key = f"{pa}_vs_{pb}"
        if pa in curves and pb in curves:
            a, b = curves[pa], curves[pb]
            n = min(len(a), len(b))
            if n >= 3:
                r, p = _pearsonr(a[:n], b[:n])
                out[key] = {"r": round(float(r), 4), "p": round(float(p), 6)}
            else:
                out[key] = None
        else:
            out[key] = None
    return out


def _save_pearson_static(all_data: dict, model_short: str, cfg: dict, slug: str) -> None:
    """Save Pearson correlation heatmap table as PNG + PDF."""
    if not _HAS_MPL or not _HAS_NUMPY:
        print("  [static] matplotlib/numpy not available — skipping PNG/PDF")
        return
    modules = [m for m in cfg["modules"] if m in all_data]
    corr_pairs = cfg["corr_pairs"]
    n_mods  = len(modules)
    n_pairs = len(corr_pairs)
    r_mat = np.full((n_mods, n_pairs), np.nan)
    texts = [["N/A"] * n_pairs for _ in range(n_mods)]
    for i, mod in enumerate(modules):
        corrs = all_data[mod]["correlations"]
        for j, (pa, pb) in enumerate(corr_pairs):
            entry = corrs.get(f"{pa}_vs_{pb}")
            if entry:
                r_mat[i, j] = entry["r"]
                p = entry["p"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                texts[i][j] = f"{entry['r']:.3f}\n{sig}"
    col_labels = [f"{pa.replace('_','\u2192')}\nvs\n{pb.replace('_','\u2192')}"
                  for pa, pb in corr_pairs]
    _mpl_pub_style()
    fig, ax = plt.subplots(figsize=(n_pairs * 2.0 + 1.2, n_mods * 1.5 + 2.0))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#d0d0d0")
    im = ax.imshow(r_mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n_mods))
    ax.set_yticklabels(modules, fontsize=10, fontweight="bold")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for i in range(n_mods):
        for j in range(n_pairs):
            r_val = r_mat[i, j]
            fc = "white" if (not np.isnan(r_val) and abs(r_val) > 0.6) else "black"
            ax.text(j, i, texts[i][j], ha="center", va="center",
                    fontsize=9, color=fc, fontweight="bold", linespacing=1.4)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Pearson r")
    pos = cfg.get("position", "answer_colon")
    ax.set_title(
        f"KL Divergence Pearson Correlation — {model_short}\nPosition: {pos}",
        fontsize=12, pad=32)
    fig.tight_layout()
    _save_fig(fig, slug)
    plt.close(fig)


def _save_pearson_curves_static(module_data: dict, model_short: str,
                                 cfg: dict, slug: str) -> None:
    """Save per-layer KL divergence line plots as PNG + PDF."""
    if not _HAS_MPL:
        print("  [static] matplotlib not available — skipping PNG/PDF")
        return
    modules = [m for m in cfg["modules"] if m in module_data]
    if not modules:
        return
    _COLORS = {"en_ko": "#1565c0", "en_zh": "#2e7d32", "ko_zh": "#b71c1c",
               "ko_en": "#6a1b9a", "en_ja": "#e65100", "ko_ja": "#37474f"}
    _mpl_pub_style()
    n = len(modules)
    fig, axes = plt.subplots(1, n, figsize=(max(5, 5 * n), 3.5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, module in zip(axes, modules):
        curves = module_data[module]
        for pair, vals in curves.items():
            ax.plot(vals, label=pair.replace("_", "\u2192"),
                    color=_COLORS.get(pair, "#546e7a"), linewidth=1.8)
        ax.set_xlabel("Layer", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("KL Divergence (avg)", fontsize=10)
        ax.set_title(module.upper(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    pos = cfg.get("position", "answer_colon")
    fig.suptitle(f"Per-Layer KL Divergence — {model_short}  [{pos}]",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, slug)
    plt.close(fig)


def _run_pearson(cfg: dict) -> None:
    print("[step2_5_pearson] Loading data ...")
    for model in cfg["models"]:
        print(f"  Model: {model}")
        module_data = _load_pearson_data(cfg, model)
        if not module_data:
            print(f"  No data for {model} — skipping")
            continue
        all_data: dict = {}
        for module, curves in module_data.items():
            corrs = _compute_correlations(curves, cfg["corr_pairs"])
            all_data[module] = {"curves": curves, "correlations": corrs}
        slug = f"step2_5_pearson_{model}"
        # Interactive HTML
        html = _inject(_PEARSON_HTML, {
            "DATA_JSON":       _embed_json(all_data),
            "CORR_PAIRS_JSON": _embed_json(cfg["corr_pairs"]),
            "MODEL_SHORT":     model,
            "POSITION":        cfg.get("position", "answer_colon"),
        })
        _save_html(html, slug)
        # Publication PNG + PDF
        _save_pearson_static(all_data, model, cfg, slug)
        _save_pearson_curves_static(module_data, model, cfg, f"{slug}_curves")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate visualizations for multilingual_hop pipeline outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--step5_stage2_5_attention", action="store_true",
                   help="Attention heatmap (stage2.5) — HTML only")
    p.add_argument("--step4_bhs", action="store_true",
                   help="Bridge Head Score layer x head heatmap (step4) — HTML + PNG/PDF")
    p.add_argument("--step5_stage2_ablation", action="store_true",
                   help="Mean ablation delta bar chart (stage2) — HTML + PNG/PDF")
    p.add_argument("--step2_5_pearson", action="store_true",
                   help="Activation patching KL Pearson correlation table — HTML + PNG/PDF")
    p.add_argument("--all", action="store_true",
                   help="Run all visualizations")
    args = p.parse_args()

    any_selected = (
        args.step5_stage2_5_attention
        or args.step4_bhs
        or args.step5_stage2_ablation
        or args.step2_5_pearson
    )
    run_all = args.all or not any_selected

    print(f"Output directory: {OUTPUT_DIR}\n")

    if run_all or args.step5_stage2_5_attention:
        _run_attention(CONFIGS["step5_stage2_5_attention"])

    if run_all or args.step4_bhs:
        _run_bhs(CONFIGS["step4_bhs"])

    if run_all or args.step5_stage2_ablation:
        _run_ablation(CONFIGS["step5_stage2_ablation"])

    if run_all or args.step2_5_pearson:
        _run_pearson(CONFIGS["step2_5_pearson"])

    print("\nDone.")


if __name__ == "__main__":
    main()
