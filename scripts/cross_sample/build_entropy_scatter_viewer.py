"""
Build a standalone HTML viewer for the JS-threshold entropy scatter plots.

Reads the summary written by plot_entropy_above_js_linear.py, embeds every PNG
as base64, and emits a single self-contained page with a slider that steps
through thresholds from low to high. No web server needed - open the file in a
browser (or copy it to your laptop; it has no external dependencies).

Edit the global parameters below, then run:
  python build_entropy_scatter_viewer.py
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

RESULT_ROOT = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/"
    "cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js"
)

SUMMARY_NAME = "scatter_linear_summary.json"
OUTPUT_NAME = "entropy_scatter_viewer.html"

PAGE_TITLE = "Student / Teacher entropy vs JS threshold"


# ============================================================================ #
#                                  Implementation                              #
# ============================================================================ #

import base64
import json
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; }
  body {
    margin: 0;
    padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f6f8;
    color: #1b1f24;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
  .sub { font-size: 13px; color: #5a626b; margin-bottom: 20px; }
  .panel {
    background: #fff;
    border: 1px solid #e0e3e7;
    border-radius: 10px;
    padding: 20px;
  }
  .layout { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
  .figure { flex: 1 1 560px; min-width: 320px; text-align: center; }
  .figure img {
    width: 100%;
    max-width: 620px;
    image-rendering: auto;
    border-radius: 6px;
  }
  .axis-note { font-size: 12px; color: #5a626b; margin-top: 8px; }
  .side { flex: 0 1 320px; min-width: 260px; }
  .thval {
    font-size: 34px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    margin-bottom: 2px;
  }
  .thlabel { font-size: 12px; text-transform: uppercase; letter-spacing: .07em; color: #5a626b; }
  input[type=range] { width: 100%; margin: 18px 0 6px; accent-color: #2f80c4; }
  .ticks { display: flex; justify-content: space-between; font-size: 11px; color: #7b838c; }
  table { width: 100%; border-collapse: collapse; margin-top: 18px; font-size: 13px; }
  td { padding: 6px 0; border-bottom: 1px solid #eef0f2; }
  td.k { color: #5a626b; }
  td.v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .quad {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 16px;
  }
  .cell {
    background: #f0f7fc;
    border: 1px solid #d7e7f3;
    border-radius: 6px;
    padding: 8px 10px;
  }
  .cell .n { font-size: 15px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .cell .l { font-size: 10px; color: #5a626b; text-transform: uppercase; letter-spacing: .05em; }
  .controls { display: flex; gap: 8px; margin-top: 18px; flex-wrap: wrap; }
  button {
    font: inherit;
    font-size: 13px;
    padding: 7px 14px;
    border: 1px solid #cfd4da;
    border-radius: 6px;
    background: #fff;
    cursor: pointer;
  }
  button:hover { background: #eef1f4; }
  button.on { background: #2f80c4; border-color: #2f80c4; color: #fff; }
  .hint { font-size: 12px; color: #7b838c; margin-top: 14px; line-height: 1.5; }
  kbd {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
    background: #eef1f4;
    border: 1px solid #d7dbe0;
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 1px 5px;
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">
    x = student top-__TOPK__ entropy, y = teacher top-__TOPK__ entropy, over tokens with JS &gt; threshold.
    Both axes linear on [__AXMIN__, __AXMAX__] (nats); dashed lines at __SPLIT__.
  </div>

  <div class="panel">
    <div class="layout">
      <div class="figure">
        <img id="plot" alt="entropy scatter">
        <div class="axis-note" id="imgpath"></div>
      </div>

      <div class="side">
        <div class="thlabel">JS threshold</div>
        <div class="thval" id="thval"></div>
        <input type="range" id="slider" min="0" max="0" step="1" value="0">
        <div class="ticks"><span id="tickmin"></span><span id="tickmax"></span></div>

        <table>
          <tr><td class="k">Total response tokens</td><td class="v" id="s-total"></td></tr>
          <tr><td class="k">Selected (JS &gt; th)</td><td class="v" id="s-sel"></td></tr>
          <tr><td class="k">Selected ratio</td><td class="v" id="s-ratio"></td></tr>
          <tr><td class="k">Plotted points</td><td class="v" id="s-plotted"></td></tr>
          <tr><td class="k">Pearson r</td><td class="v" id="s-corr"></td></tr>
          <tr><td class="k">Mean student entropy</td><td class="v" id="s-ms"></td></tr>
          <tr><td class="k">Mean teacher entropy</td><td class="v" id="s-mt"></td></tr>
        </table>

        <div class="quad">
          <div class="cell"><div class="n" id="q-lh"></div><div class="l">S low / T high</div></div>
          <div class="cell"><div class="n" id="q-hh"></div><div class="l">S high / T high</div></div>
          <div class="cell"><div class="n" id="q-ll"></div><div class="l">S low / T low</div></div>
          <div class="cell"><div class="n" id="q-hl"></div><div class="l">S high / T low</div></div>
        </div>

        <div class="controls">
          <button id="prev">&larr; Prev</button>
          <button id="next">Next &rarr;</button>
          <button id="play">Play</button>
        </div>

        <div class="hint">
          <kbd>&larr;</kbd> <kbd>&rarr;</kbd> step thresholds. Images are embedded,
          so switching is instant and the file works offline.
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const SPLIT = __SPLIT__;

const el = (id) => document.getElementById(id);
const slider = el("slider");
const fmtInt = (n) => n.toLocaleString("en-US");
const fmtPct = (x) => (100 * x).toFixed(2) + "%";
const fmtNum = (x, d = 4) => (x === null || Number.isNaN(x)) ? "n/a" : x.toFixed(d);

slider.max = String(DATA.length - 1);
el("tickmin").textContent = "th = " + DATA[0].js_threshold;
el("tickmax").textContent = "th = " + DATA[DATA.length - 1].js_threshold;

function render(i) {
  const d = DATA[i];
  el("plot").src = d.image_data;
  el("thval").textContent = d.js_threshold;
  el("imgpath").textContent = d.image_name;
  el("s-total").textContent = fmtInt(d.total_tokens);
  el("s-sel").textContent = fmtInt(d.selected_tokens);
  el("s-ratio").textContent = fmtPct(d.selected_ratio);
  el("s-plotted").textContent = fmtInt(d.plotted_points);
  el("s-corr").textContent = fmtNum(d.correlation);
  el("s-ms").textContent = fmtNum(d.mean_student_entropy, 3);
  el("s-mt").textContent = fmtNum(d.mean_teacher_entropy, 3);
  el("q-ll").textContent = fmtInt(d.quadrants.low_low);
  el("q-lh").textContent = fmtInt(d.quadrants.low_high);
  el("q-hl").textContent = fmtInt(d.quadrants.high_low);
  el("q-hh").textContent = fmtInt(d.quadrants.high_high);
}

function step(delta) {
  const next = Math.min(DATA.length - 1, Math.max(0, Number(slider.value) + delta));
  slider.value = String(next);
  render(next);
}

slider.addEventListener("input", () => render(Number(slider.value)));
el("prev").addEventListener("click", () => step(-1));
el("next").addEventListener("click", () => step(1));

document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") { step(-1); event.preventDefault(); }
  if (event.key === "ArrowRight") { step(1); event.preventDefault(); }
});

let timer = null;
el("play").addEventListener("click", () => {
  const button = el("play");
  if (timer) {
    clearInterval(timer);
    timer = null;
    button.textContent = "Play";
    button.classList.remove("on");
    return;
  }
  button.textContent = "Pause";
  button.classList.add("on");
  timer = setInterval(() => {
    const current = Number(slider.value);
    const next = current >= DATA.length - 1 ? 0 : current + 1;
    slider.value = String(next);
    render(next);
  }, 900);
});

render(0);
</script>
</body>
</html>
"""


def main():
    root = Path(RESULT_ROOT).resolve()
    summary_path = root / SUMMARY_NAME
    output_path = root / OUTPUT_NAME

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Summary not found: {summary_path}\n"
            "Run plot_entropy_above_js_linear.py first."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = sorted(summary["entries"], key=lambda item: item["js_threshold"])
    if not entries:
        raise ValueError(f"No entries in {summary_path}")

    payload = []
    for entry in entries:
        image_path = Path(entry["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing figure: {image_path}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload.append({
            "js_threshold": entry["js_threshold"],
            "image_name": image_path.name,
            "image_data": f"data:image/png;base64,{encoded}",
            "total_tokens": entry["total_tokens"],
            "selected_tokens": entry["selected_tokens"],
            "selected_ratio": entry["selected_ratio"],
            "plotted_points": entry["plotted_points"],
            "correlation": entry["correlation"],
            "mean_student_entropy": entry["mean_student_entropy"],
            "mean_teacher_entropy": entry["mean_teacher_entropy"],
            "quadrants": entry["quadrants"],
        })

    html = HTML_TEMPLATE
    for token, value in (
        ("__TITLE__", PAGE_TITLE),
        ("__TOPK__", str(summary["top_k"])),
        ("__AXMIN__", str(summary["axis_min"])),
        ("__AXMAX__", str(summary["axis_max"])),
        ("__SPLIT__", str(summary["split_line"])),
        ("__DATA__", json.dumps(payload)),
    ):
        html = html.replace(token, value)

    output_path.write_text(html, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Embedded {len(payload)} thresholds: "
          f"{entries[0]['js_threshold']} -> {entries[-1]['js_threshold']}")
    print(f"Wrote {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
