"""Step 2.5 Extra: Pearson correlation matrix across language-pair patching profiles.

For each model × module, load the 20 average_*.json files (5 langs × 4 pairs each),
extract the layer-wise KL and LD vectors at the 'answer_colon' position, and compute
pairwise Pearson correlations → 20×20 heatmap.

Also computes N-variable concordance measures:
  - Kendall's W (coefficient of concordance over all 20 rankings)
  - PCA explained variance ratio (how much variance is shared across pairs)
  - Mean pairwise Pearson r

Output: output/extra/step2_5_pearson_corr/{model}/{module}/
  correlation_kl.csv, correlation_ld.csv, report.html

Usage:
  python script/extra/step2_5_pearson_correlation.py \
      --models llama31_70 qwen25_72 \
      --modules attn mlp
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import permutations
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANGS = ["en", "ko", "zh", "ja", "es"]
DEFAULT_MODELS = ["llama31_70", "qwen25_72"]
DEFAULT_MODULES = ["attn", "mlp"]

PATCHING_ROOT = PROJECT_ROOT / "output" / "step2_5_activation_patching"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "extra" / "step2_5_pearson_corr"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class PairProfile(NamedTuple):
    lang_a: str
    lang_b: str
    kl: list[float]   # length = n_layers
    ld: list[float]   # length = n_layers
    n_layers: int
    num_samples: int


def load_pair_profiles(
    model: str,
    module: str,
    langs: list[str],
) -> list[PairProfile]:
    base = PATCHING_ROOT / model / module
    profiles: list[PairProfile] = []
    missing: list[str] = []
    for a, b in permutations(langs, 2):
        path = base / f"average_{a}_{b}.json"
        if not path.exists():
            missing.append(f"{a}_{b}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        kl = data["per_layer_kl"]["answer_colon"]
        ld = data["per_layer_logit_diff"]["answer_colon"]
        profiles.append(PairProfile(
            lang_a=a,
            lang_b=b,
            kl=kl,
            ld=ld,
            n_layers=data["n_layers"],
            num_samples=data.get("num_samples", 0),
        ))
    if missing:
        print(f"  [warn] {model}/{module}: missing pairs: {missing}")
    return profiles


# ---------------------------------------------------------------------------
# Pure-Python statistics (no numpy/scipy required at import time)
# ---------------------------------------------------------------------------
def _mean(v: list[float]) -> float:
    return sum(v) / len(v)


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = _mean(x), _mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx < 1e-14 or dy < 1e-14:
        return float("nan")
    r = num / (dx * dy)
    return max(-1.0, min(1.0, r))


def build_corr_matrix(
    profiles: list[PairProfile],
    metric: str,  # "kl" | "ld"
) -> list[list[float]]:
    n = len(profiles)
    vecs = [getattr(p, metric) for p in profiles]
    mat: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = _pearson(vecs[i], vecs[j])
    return mat


# ---------------------------------------------------------------------------
# N-variable concordance measures
# ---------------------------------------------------------------------------
def kendalls_w(profiles: list[PairProfile], metric: str) -> float:
    """Kendall's W (coefficient of concordance) over N layer-profiles.

    Treat each profile as a 'judge' ranking the n_layers 'subjects'.
    W = 1 means all profiles rank layers identically; W = 0 means no agreement.
    """
    vecs = [getattr(p, metric) for p in profiles]
    if not vecs:
        return float("nan")
    n_judges = len(vecs)
    n_items = len(vecs[0])
    # Convert each vector to ranks (1-based, average for ties)
    def _rank(v: list[float]) -> list[float]:
        sorted_idx = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j < len(v) - 1 and v[sorted_idx[j + 1]] == v[sorted_idx[j]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks
    rank_matrix = [_rank(v) for v in vecs]
    # Column sums of ranks
    col_sums = [sum(rank_matrix[j][i] for j in range(n_judges)) for i in range(n_items)]
    mean_col = sum(col_sums) / n_items
    ss = sum((s - mean_col) ** 2 for s in col_sums)
    # Max possible S
    s_max = (n_judges ** 2) * (n_items ** 3 - n_items) / 12
    if s_max < 1e-14:
        return float("nan")
    return ss / s_max


def pca_first_pc_var(profiles: list[PairProfile], metric: str) -> float:
    """Fraction of total variance explained by the first principal component.

    Uses pure Python via the power-iteration method.
    High value → all language pairs share a single dominant layer-importance pattern.
    """
    vecs = [getattr(p, metric) for p in profiles]
    n, d = len(vecs), len(vecs[0])
    if n < 2 or d < 2:
        return float("nan")
    # Center each feature (layer) across the n profiles
    col_means = [sum(vecs[i][j] for i in range(n)) / n for j in range(d)]
    centered = [[vecs[i][j] - col_means[j] for j in range(d)] for i in range(n)]
    # Covariance matrix (d×d): C = X^T X / (n-1)
    # Power iteration to find largest eigenvalue
    import random
    rng = random.Random(0)
    vec = [rng.gauss(0, 1) for _ in range(d)]
    for _ in range(200):
        # multiply C * vec  = X^T (X * vec) / (n-1)
        # step 1: X * vec  (shape n)
        xv = [sum(centered[i][j] * vec[j] for j in range(d)) for i in range(n)]
        # step 2: X^T * xv (shape d)
        xtxv = [sum(centered[i][j] * xv[i] for i in range(n)) for j in range(d)]
        norm = math.sqrt(sum(x * x for x in xtxv))
        if norm < 1e-14:
            return float("nan")
        vec = [x / norm for x in xtxv]
    # Rayleigh quotient: eigenvalue estimate
    xv = [sum(centered[i][j] * vec[j] for j in range(d)) for i in range(n)]
    xtxv = [sum(centered[i][j] * xv[i] for i in range(n)) for j in range(d)]
    lambda1 = sum(vec[j] * xtxv[j] for j in range(d)) / (n - 1)
    # Total variance = sum of diagonal of C
    total_var = sum(
        sum((centered[i][j] ** 2) for i in range(n)) / (n - 1)
        for j in range(d)
    )
    if total_var < 1e-14:
        return float("nan")
    return lambda1 / total_var


def mean_pairwise_pearson(mat: list[list[float]]) -> float:
    n = len(mat)
    vals = [mat[i][j] for i in range(n) for j in range(n) if i != j and not math.isnan(mat[i][j])]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def save_csv(profiles: list[PairProfile], mat: list[list[float]], path: Path) -> None:
    labels = [f"{p.lang_a}→{p.lang_b}" for p in profiles]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, row in enumerate(mat):
            cells = ",".join(f"{v:.4f}" if not math.isnan(v) else "nan" for v in row)
            f.write(f"{labels[i]},{cells}\n")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
_LANG_COLORS = {
    "en": "#3b82f6",
    "ko": "#f59e0b",
    "zh": "#10b981",
    "ja": "#ef4444",
    "es": "#8b5cf6",
}


def _rgb_for_val(v: float) -> str:
    """Map Pearson r ∈ [-1,1] → blue-white-red colormap."""
    if math.isnan(v):
        return "rgb(200,200,200)"
    t = (v + 1.0) / 2.0  # 0=blue, 0.5=white, 1=red
    if t < 0.5:
        s = t * 2.0
        r = int(220 * s + 239 * (1 - s))
        g = int(220 * s + 246 * (1 - s))
        b = int(220 * s + 253 * (1 - s))
        # Actually do proper blue→white
        r = int(255 * s + 59 * (1 - s))
        g = int(255 * s + 130 * (1 - s))
        b = int(255 * s + 246 * (1 - s))
    else:
        s = (t - 0.5) * 2.0
        r = int(252 * s + 255 * (1 - s))
        g = int(100 * s + 255 * (1 - s))
        b = int(100 * s + 255 * (1 - s))
    return f"rgb({r},{g},{b})"


def build_heatmap_html(
    profiles: list[PairProfile],
    mat: list[list[float]],
    metric_label: str,
    title: str,
) -> str:
    n = len(profiles)
    labels = [f"{p.lang_a}→{p.lang_b}" for p in profiles]

    # Build cell data as JSON for interactive tooltip
    cell_data: list[dict] = []
    for i in range(n):
        for j in range(n):
            cell_data.append({
                "i": i, "j": j,
                "r": round(mat[i][j], 4) if not math.isnan(mat[i][j]) else None,
                "la_i": profiles[i].lang_a, "lb_i": profiles[i].lang_b,
                "la_j": profiles[j].lang_a, "lb_j": profiles[j].lang_b,
            })

    cells_json = json.dumps(cell_data, ensure_ascii=False)
    labels_json = json.dumps(labels)
    colors = [_rgb_for_val(mat[i][j]) for i in range(n) for j in range(n)]
    colors_json = json.dumps(colors)
    values_json = json.dumps([
        round(mat[i][j], 4) if not math.isnan(mat[i][j]) else None
        for i in range(n) for j in range(n)
    ])

    lang_color_js = json.dumps(_LANG_COLORS)

    return f"""<div class="matrix-block">
<h3 class="matrix-title">{title}</h3>
<div class="matrix-scroll">
<canvas id="{metric_label}_canvas" class="matrix-canvas" width="{n * 30 + 80}" height="{n * 30 + 80}"></canvas>
</div>
<div id="{metric_label}_tooltip" class="tooltip" style="display:none"></div>
<script>
(function(){{
  const LABELS = {labels_json};
  const COLORS = {colors_json};
  const VALUES = {values_json};
  const CELLS = {cell_data_json(cell_data)};
  const LANG_COLORS = {lang_color_js};
  const N = {n};
  const CELL = 30;
  const PAD_L = 80, PAD_T = 80;
  const cv = document.getElementById('{metric_label}_canvas');
  const ctx = cv.getContext('2d');
  const W = N * CELL + PAD_L, H = N * CELL + PAD_T;
  cv.width = W; cv.height = H;
  cv.style.width = W + 'px'; cv.style.height = H + 'px';

  function drawMatrix() {{
    ctx.clearRect(0, 0, W, H);
    ctx.font = '9px "IBM Plex Mono",monospace';
    // Row labels
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < N; i++) {{
      const lbl = LABELS[i];
      const parts = lbl.split('→');
      ctx.fillStyle = LANG_COLORS[parts[0]] || '#555';
      ctx.fillText(lbl, PAD_L - 4, PAD_T + i * CELL + CELL / 2);
    }}
    // Column labels (rotated)
    ctx.save();
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (let j = 0; j < N; j++) {{
      const lbl = LABELS[j];
      const parts = lbl.split('→');
      ctx.save();
      ctx.translate(PAD_L + j * CELL + CELL / 2, PAD_T - 4);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = LANG_COLORS[parts[0]] || '#555';
      ctx.fillText(lbl, 0, 0);
      ctx.restore();
    }}
    ctx.restore();
    // Cells
    for (let i = 0; i < N; i++) {{
      for (let j = 0; j < N; j++) {{
        const idx = i * N + j;
        ctx.fillStyle = COLORS[idx];
        ctx.fillRect(PAD_L + j * CELL, PAD_T + i * CELL, CELL - 1, CELL - 1);
        const v = VALUES[idx];
        if (v !== null) {{
          ctx.fillStyle = Math.abs(v) > 0.5 ? '#fff' : '#333';
          ctx.font = '7px "IBM Plex Mono",monospace';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(v.toFixed(2), PAD_L + j * CELL + CELL / 2, PAD_T + i * CELL + CELL / 2);
        }}
      }}
    }}
    // Color scale
    const barX = PAD_L + N * CELL + 10, barY = PAD_T, barH = N * CELL, barW = 14;
    for (let k = 0; k < barH; k++) {{
      const t = 1 - k / barH;
      const v = t * 2 - 1;
      const r = v >= 0
        ? [Math.round(252 * v + 255 * (1-v)), Math.round(100 * v + 255 * (1-v)), Math.round(100 * v + 255 * (1-v))]
        : [Math.round(255 * (1+v) + 59 * (-v)), Math.round(255 * (1+v) + 130 * (-v)), Math.round(255 * (1+v) + 246 * (-v))];
      ctx.fillStyle = `rgb(${{r[0]}},${{r[1]}},${{r[2]}})`;
      ctx.fillRect(barX, barY + k, barW, 1);
    }}
    ctx.fillStyle = '#333';
    ctx.font = '9px "IBM Plex Mono",monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('+1', barX + barW + 3, barY);
    ctx.textBaseline = 'middle';
    ctx.fillText(' 0', barX + barW + 3, barY + barH / 2);
    ctx.textBaseline = 'bottom';
    ctx.fillText('-1', barX + barW + 3, barY + barH);
  }}

  drawMatrix();

  // Hover tooltip
  const tip = document.getElementById('{metric_label}_tooltip');
  cv.addEventListener('mousemove', e => {{
    const rect = cv.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (W / rect.width) - PAD_L;
    const y = (e.clientY - rect.top) * (H / rect.height) - PAD_T;
    const j = Math.floor(x / CELL), i = Math.floor(y / CELL);
    if (i >= 0 && i < N && j >= 0 && j < N) {{
      const v = VALUES[i * N + j];
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top = (e.clientY - 28) + 'px';
      tip.innerHTML = `<b>${{LABELS[i]}}</b> vs <b>${{LABELS[j]}}</b><br>r = ${{v !== null ? v.toFixed(4) : 'n/a'}}`;
    }} else {{
      tip.style.display = 'none';
    }}
  }});
  cv.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
}})();
</script>
</div>"""


def cell_data_json(cells: list[dict]) -> str:
    return json.dumps(cells, ensure_ascii=False)


def build_report_html(
    model: str,
    module: str,
    profiles: list[PairProfile],
    kl_mat: list[list[float]],
    ld_mat: list[list[float]],
    stats: dict,
) -> str:
    stats_json = json.dumps(stats, ensure_ascii=False, indent=2)

    kl_block = build_heatmap_html(profiles, kl_mat, f"{model}_{module}_kl", "KL Divergence — answer_colon position")
    ld_block = build_heatmap_html(profiles, ld_mat, f"{model}_{module}_ld", "Logit Difference — answer_colon position")

    pair_table_rows = ""
    for p in profiles:
        pair_table_rows += (
            f"<tr><td>{p.lang_a}→{p.lang_b}</td>"
            f"<td>{p.num_samples}</td>"
            f"<td>{p.n_layers}</td></tr>\n"
        )

    kl_w = stats.get("kl_kendalls_w", float("nan"))
    ld_w = stats.get("ld_kendalls_w", float("nan"))
    kl_pca = stats.get("kl_pca_pc1_var", float("nan"))
    ld_pca = stats.get("ld_pca_pc1_var", float("nan"))
    kl_mean_r = stats.get("kl_mean_pairwise_r", float("nan"))
    ld_mean_r = stats.get("ld_mean_pairwise_r", float("nan"))

    def fmt(v):
        return f"{v:.4f}" if not (v is None or (isinstance(v, float) and math.isnan(v))) else "n/a"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pearson Correlation | {model} | {module}</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#fafafa;--surface:#fff;--border:#e2e5ea;
  --text:#1c2333;--muted:#6b7585;--blue:#2563eb;
  --radius:8px;
  --font:'IBM Plex Sans','Helvetica Neue',Arial,sans-serif;
  --mono:'IBM Plex Mono','Fira Code',monospace;
}}
html{{font-size:14px}}
body{{font-family:var(--font);background:var(--bg);color:var(--text);line-height:1.5}}
.page{{max-width:1200px;margin:0 auto;padding:28px 20px 80px}}
.hdr{{border-bottom:1.5px solid var(--border);padding-bottom:18px;margin-bottom:24px}}
.hdr h1{{font-size:1.15rem;font-weight:600;margin-bottom:6px}}
.hdr .sub{{font-size:.78rem;color:var(--muted);font-family:var(--mono)}}
.badge{{display:inline-block;padding:2px 9px;border-radius:4px;font-size:.7rem;font-weight:600;
        letter-spacing:.05em;text-transform:uppercase;background:#dbeafe;color:#1e40af;margin-left:8px}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
       margin-bottom:24px;overflow:hidden}}
.card-head{{display:flex;align-items:baseline;gap:10px;padding:12px 18px 10px;
            border-bottom:1px solid var(--border)}}
.card-head h2{{font-size:.88rem;font-weight:600}}
.card-body{{padding:20px 18px}}

/* stats table */
.stats-table{{border-collapse:collapse;width:100%;max-width:600px;font-size:.8rem}}
.stats-table th,.stats-table td{{padding:6px 14px;text-align:left;
  border-bottom:1px solid var(--border)}}
.stats-table th{{color:var(--muted);font-weight:500}}
.stats-table td.mono{{font-family:var(--mono);color:var(--blue)}}
.stats-table .section{{background:#f8fafc;font-weight:600;color:var(--text)}}

/* matrix */
.matrix-block{{margin-bottom:32px}}
.matrix-title{{font-size:.82rem;font-weight:600;margin-bottom:10px;color:var(--muted)}}
.matrix-scroll{{overflow-x:auto}}
.matrix-canvas{{display:block;cursor:crosshair}}

/* tooltip */
.tooltip{{position:fixed;background:rgba(28,35,51,.92);color:#fff;
          font-size:.72rem;font-family:var(--mono);padding:7px 11px;
          border-radius:5px;pointer-events:none;z-index:9999;line-height:1.6}}

/* interpretation box */
.interp{{background:#f0f9ff;border:1px solid #bae6fd;border-radius:var(--radius);
         padding:16px 20px;font-size:.8rem;line-height:1.7;margin-bottom:24px}}
.interp h3{{font-size:.85rem;font-weight:600;margin-bottom:8px;color:#0369a1}}
.interp ul{{padding-left:18px}}
.interp li{{margin-bottom:4px}}
.highlight{{background:#fef9c3;padding:1px 4px;border-radius:3px}}

/* pair table */
.pair-table{{border-collapse:collapse;font-size:.75rem}}
.pair-table th,.pair-table td{{padding:3px 10px;text-align:left;
  border-bottom:1px solid var(--border)}}
.pair-table th{{color:var(--muted);font-weight:500}}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
</head>
<body>
<div class="page">

<div class="hdr">
  <h1>Pearson Correlation of Layer-wise Patching Profiles
    <span class="badge">{model}</span>
    <span class="badge" style="background:#dcfce7;color:#166534">{module}</span>
  </h1>
  <div class="sub">
    Position: <b>answer_colon</b> &nbsp;|&nbsp;
    Metric: layer-wise KL divergence &amp; logit difference &nbsp;|&nbsp;
    {len(profiles)} language pairs
  </div>
</div>

<!-- Interpretation -->
<div class="interp">
  <h3>How to interpret KL vs Logit Difference (LD)</h3>
  <ul>
    <li><b>KL divergence (per layer):</b> When layer L's activation is patched from language A into
        language B's forward pass, how much does the model's output <em>distribution</em> shift?
        <span class="highlight">High KL at layer L → that layer is causally important for cross-lingual knowledge transfer.</span>
        The layer with peak KL is where the model "bridges" between the source and target language representation.</li>
    <li><b>Logit Difference (LD, per layer):</b> Change in the <em>correct token's logit</em> after patching layer L.
        <span class="highlight">Positive LD → patching brought useful source-language information into the target path
        (the correct answer becomes more likely).</span>
        Negative LD → patching overwrote useful target-language context.</li>
    <li><b>attn module:</b> Attention outputs carry positional/relational routing. High KL in early attention layers
        suggests the model uses attention to identify which language to "read" the answer from.</li>
    <li><b>mlp module:</b> MLPs store factual knowledge. High KL in mid-to-late MLP layers indicates where
        factual bridging entities are resolved.</li>
    <li><b>Llama vs Qwen (attn/mlp comparison):</b> If Llama shows earlier KL peaks in attn but later MLP peaks
        compared to Qwen, it suggests different cross-lingual routing strategies — Llama may route language identity
        earlier while Qwen consolidates it in deeper MLP layers.</li>
  </ul>
</div>

<!-- N-variable concordance statistics -->
<div class="card">
  <div class="card-head"><h2>N-variable Concordance Statistics</h2></div>
  <div class="card-body">
    <table class="stats-table">
      <tr><th>Statistic</th><th>KL (answer_colon)</th><th>LD (answer_colon)</th><th>Interpretation</th></tr>
      <tr class="section"><td colspan="4">Pairwise Pearson Summary</td></tr>
      <tr>
        <td>Mean pairwise Pearson r</td>
        <td class="mono">{fmt(kl_mean_r)}</td>
        <td class="mono">{fmt(ld_mean_r)}</td>
        <td>Average correlation between any two lang-pair profiles</td>
      </tr>
      <tr class="section"><td colspan="4">Kendall's W — Coefficient of Concordance</td></tr>
      <tr>
        <td>Kendall's W</td>
        <td class="mono">{fmt(kl_w)}</td>
        <td class="mono">{fmt(ld_w)}</td>
        <td>W=1: all pairs agree on layer ranking; W=0: no agreement.<br>
            Extends Pearson to N≥2 variables simultaneously.</td>
      </tr>
      <tr class="section"><td colspan="4">PCA — Principal Component Analysis</td></tr>
      <tr>
        <td>PC1 explained variance</td>
        <td class="mono">{fmt(kl_pca)}</td>
        <td class="mono">{fmt(ld_pca)}</td>
        <td>Fraction of variance in the {len(profiles)} layer-profiles captured by one shared pattern.
            High value → a single dominant cross-lingual patching curve.</td>
      </tr>
    </table>

    <div style="margin-top:18px;font-size:.78rem;color:var(--muted);max-width:680px;line-height:1.7">
      <b>N-variable correlation methods beyond pairwise Pearson:</b><br>
      <b>1. Kendall's W</b> generalizes rank correlation to N≥2 variables simultaneously —
      it measures whether all N profiles agree on the <em>ordering</em> of layers.<br>
      <b>2. PCA explained variance</b> captures <em>linear</em> shared structure: if PC1 explains &gt;80%,
      one universal "patching curve" dominates all language pairs.<br>
      <b>3. RV coefficient</b> (Robert &amp; Escoufier 1976) generalizes Pearson to two <em>matrices</em>:
      RV(X,Y) = tr(X'YY'X) / √[tr(X'XX'X)·tr(Y'YY'Y)] — useful for comparing attn vs mlp correlation matrices.<br>
      <b>4. Multiple correlation R</b>: correlation between one pair's profile and the linear combination
      of all other pairs' profiles that best predicts it.<br>
      <b>5. CCA</b> (Canonical Correlation Analysis): finds paired linear combinations of two sets of
      language-pair profiles that maximize mutual correlation.
    </div>
  </div>
</div>

<!-- Loaded pairs -->
<div class="card">
  <div class="card-head"><h2>Language Pairs Loaded</h2></div>
  <div class="card-body">
    <table class="pair-table">
      <tr><th>Pair</th><th>Samples (avg)</th><th>Layers</th></tr>
      {pair_table_rows}
    </table>
  </div>
</div>

<!-- Heatmaps -->
<div class="card">
  <div class="card-head"><h2>Pearson Correlation Heatmaps</h2></div>
  <div class="card-body">
    {kl_block}
    {ld_block}
  </div>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# RV coefficient (attn vs mlp module comparison)
# ---------------------------------------------------------------------------
def rv_coefficient(mat_a: list[list[float]], mat_b: list[list[float]]) -> float:
    """RV coefficient between two square symmetric correlation matrices.

    Treats each matrix as an n²-vector after flattening.
    """
    n = len(mat_a)
    # Flatten, skip NaN
    a = [mat_a[i][j] for i in range(n) for j in range(n)
         if not (math.isnan(mat_a[i][j]) or math.isnan(mat_b[i][j]))]
    b = [mat_b[i][j] for i in range(n) for j in range(n)
         if not (math.isnan(mat_a[i][j]) or math.isnan(mat_b[i][j]))]
    return _pearson(a, b)


# ---------------------------------------------------------------------------
# Summary HTML (cross-model, cross-module comparison)
# ---------------------------------------------------------------------------
def build_summary_html(all_stats: list[dict]) -> str:
    rows = ""
    for s in all_stats:
        rows += (
            f"<tr>"
            f"<td><b>{s['model']}</b></td><td>{s['module']}</td>"
            f"<td class='mono'>{s.get('kl_mean_pairwise_r', float('nan')):.4f}</td>"
            f"<td class='mono'>{s.get('ld_mean_pairwise_r', float('nan')):.4f}</td>"
            f"<td class='mono'>{s.get('kl_kendalls_w', float('nan')):.4f}</td>"
            f"<td class='mono'>{s.get('ld_kendalls_w', float('nan')):.4f}</td>"
            f"<td class='mono'>{s.get('kl_pca_pc1_var', float('nan')):.4f}</td>"
            f"<td class='mono'>{s.get('ld_pca_pc1_var', float('nan')):.4f}</td>"
            f"<td><a href='{s['model']}/{s['module']}/report.html'>→ report</a></td>"
            f"</tr>\n"
        )

    # Cross-module RV coefficients
    rv_rows = ""
    by_model: dict[str, dict] = {}
    for s in all_stats:
        m = s["model"]
        mod = s["module"]
        by_model.setdefault(m, {})[mod] = s
    for model, mods in by_model.items():
        if "attn" in mods and "mlp" in mods:
            rv_kl = mods["attn"].get("_kl_mat_flat", [])
            rv_ml = mods["mlp"].get("_kl_mat_flat", [])
            if rv_kl and rv_ml and len(rv_kl) == len(rv_ml):
                rv_kl_val = _pearson(rv_kl, rv_ml)
                rv_rows += (
                    f"<tr><td><b>{model}</b></td><td>attn vs mlp</td>"
                    f"<td class='mono'>KL: {rv_kl_val:.4f}</td>"
                    f"<td>Similarity between attn and mlp KL-correlation structures</td></tr>\n"
                )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step 2.5 Pearson Correlation — Summary</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--font:'IBM Plex Sans','Helvetica Neue',Arial,sans-serif;
       --mono:'IBM Plex Mono','Fira Code',monospace;
       --border:#e2e5ea;--muted:#6b7585;--blue:#2563eb}}
html{{font-size:14px}}
body{{font-family:var(--font);background:#fafafa;color:#1c2333;line-height:1.5;
      padding:28px 24px 80px}}
h1{{font-size:1.15rem;font-weight:600;margin-bottom:6px}}
.sub{{font-size:.78rem;color:var(--muted);font-family:var(--mono);margin-bottom:24px}}
table{{border-collapse:collapse;width:100%;font-size:.8rem;margin-bottom:28px}}
th,td{{padding:7px 13px;text-align:left;border-bottom:1px solid var(--border)}}
th{{color:var(--muted);font-weight:500}}
.mono{{font-family:var(--mono);color:var(--blue)}}
a{{color:var(--blue);text-decoration:none}}
a:hover{{text-decoration:underline}}
h2{{font-size:.9rem;font-weight:600;margin:24px 0 10px}}
</style>
</head>
<body>
<h1>Step 2.5 Activation Patching — Pearson Correlation Summary</h1>
<div class="sub">Position: answer_colon &nbsp;|&nbsp; All models &amp; modules</div>

<h2>Per-model / per-module statistics</h2>
<table>
  <tr>
    <th>Model</th><th>Module</th>
    <th>KL mean r</th><th>LD mean r</th>
    <th>KL Kendall W</th><th>LD Kendall W</th>
    <th>KL PC1 var</th><th>LD PC1 var</th>
    <th>Report</th>
  </tr>
  {rows}
</table>

<h2>Cross-module structural similarity (RV / Pearson of correlation matrices)</h2>
<table>
  <tr><th>Model</th><th>Comparison</th><th>r</th><th>Interpretation</th></tr>
  {rv_rows if rv_rows else '<tr><td colspan="4" style="color:var(--muted)">Not enough modules to compare</td></tr>'}
</table>

<div style="font-size:.78rem;color:var(--muted);max-width:700px;line-height:1.7;margin-top:20px">
  <b>Interpreting the summary:</b><br>
  A high <b>Kendall's W</b> for KL means all 20 language-pair patching experiments agree on which layers matter most —
  suggesting a universal cross-lingual bridge layer.<br>
  A high <b>PC1 variance</b> corroborates this: one dominant factor explains most of the layer-profile variance.<br>
  The <b>cross-module RV</b> indicates whether attn and mlp modules show similar inter-pair correlation patterns —
  high RV means both modules transfer information in a similarly language-pair-structured way.
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 2.5 Extra: Pearson correlation of patching profiles.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--modules", nargs="+", default=DEFAULT_MODULES)
    p.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    p.add_argument(
        "--patching-root",
        type=Path,
        default=PATCHING_ROOT,
        help="Root of step2_5 output.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output directory for correlation reports.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    patching_root = args.patching_root
    output_root = args.output_root

    all_stats: list[dict] = []

    for model in args.models:
        for module in args.modules:
            print(f"\n{'='*60}")
            print(f"  Model: {model}  Module: {module}")
            print(f"{'='*60}")

            profiles = load_pair_profiles(model, module, args.langs)
            if not profiles:
                print(f"  [skip] No profiles loaded.")
                continue

            print(f"  Loaded {len(profiles)} language pairs.")

            kl_mat = build_corr_matrix(profiles, "kl")
            ld_mat = build_corr_matrix(profiles, "ld")

            # N-variable concordance
            print("  Computing Kendall's W...")
            kl_w = kendalls_w(profiles, "kl")
            ld_w = kendalls_w(profiles, "ld")
            print(f"    KL Kendall's W = {kl_w:.4f}  LD Kendall's W = {ld_w:.4f}")

            print("  Computing PCA PC1 variance...")
            kl_pca = pca_first_pc_var(profiles, "kl")
            ld_pca = pca_first_pc_var(profiles, "ld")
            print(f"    KL PC1 var = {kl_pca:.4f}  LD PC1 var = {ld_pca:.4f}")

            kl_mean_r = mean_pairwise_pearson(kl_mat)
            ld_mean_r = mean_pairwise_pearson(ld_mat)
            print(f"    KL mean r = {kl_mean_r:.4f}  LD mean r = {ld_mean_r:.4f}")

            stats = {
                "model": model,
                "module": module,
                "n_pairs": len(profiles),
                "kl_kendalls_w": kl_w,
                "ld_kendalls_w": ld_w,
                "kl_pca_pc1_var": kl_pca,
                "ld_pca_pc1_var": ld_pca,
                "kl_mean_pairwise_r": kl_mean_r,
                "ld_mean_pairwise_r": ld_mean_r,
                # Flat correlation matrix for cross-module RV
                "_kl_mat_flat": [kl_mat[i][j] for i in range(len(profiles)) for j in range(len(profiles))],
            }
            all_stats.append(stats)

            out_dir = output_root / model / module
            out_dir.mkdir(parents=True, exist_ok=True)

            # CSV
            save_csv(profiles, kl_mat, out_dir / "correlation_kl.csv")
            save_csv(profiles, ld_mat, out_dir / "correlation_ld.csv")
            print(f"  Saved CSVs → {out_dir}")

            # HTML report
            html = build_report_html(model, module, profiles, kl_mat, ld_mat, stats)
            (out_dir / "report.html").write_text(html, encoding="utf-8")
            print(f"  Saved HTML report → {out_dir / 'report.html'}")

            # JSON stats (without internal _* keys)
            public_stats = {k: v for k, v in stats.items() if not k.startswith("_")}
            (out_dir / "stats.json").write_text(
                json.dumps(public_stats, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # Summary
    if all_stats:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_html = build_summary_html(all_stats)
        (output_root / "summary.html").write_text(summary_html, encoding="utf-8")
        print(f"\nSummary saved → {output_root / 'summary.html'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
