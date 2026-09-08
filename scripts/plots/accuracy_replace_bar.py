import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ==== 配置区: 修改数据与颜色 ====
# 五种深浅不一的低饱和蓝色 (由深到浅)
BLUES = ["#3d5f83", "#4f7fa6", "#6b9dbf", "#96bcd4", "#c6dceb"]

CATEGORIES = ["ALL", "HH", "HL", "LH", "LL"]
ACCURACY = [65.00, 27.81, 47.50, 18.75, 16.25]   # 准确率 (%), 正半轴
REPLACE_RATE = [10.13, 7.19, 3.82, 1.20, 0.16]   # 替换率 (%), 负半轴

FONT_SIZE = 20
LINE_WIDTH = 3
NEG_SCALE = 2.5      # 负半轴视觉放大倍数 (仅影响柱高, 标注仍为真实值)
HIGHLIGHT_IDX = 2            # 需要强调的类别索引 (HL)
HIGHLIGHT_COLOR = "#dceaf5"  # 强调列的浅蓝色底纹
OUTPUT_NAME = "accuracy_replace_bar"
# ================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT_SIZE,
    "axes.linewidth": LINE_WIDTH,
    "axes.edgecolor": "#1A1A1A",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,
    "ytick.major.size": 6,
    "ytick.major.width": LINE_WIDTH,
    "savefig.bbox": "tight",
})

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)
x = np.arange(len(CATEGORIES))

fig, ax = plt.subplots(figsize=(8.0, 8.0))

ax.axvspan(HIGHLIGHT_IDX - 0.5, HIGHLIGHT_IDX + 0.5,
           color=HIGHLIGHT_COLOR, zorder=0)

ax.bar(x, ACCURACY, width=0.62, color=BLUES,
       edgecolor="#1A1A1A", linewidth=LINE_WIDTH, zorder=3)
ax.bar(x, [-v * NEG_SCALE for v in REPLACE_RATE], width=0.62, color=BLUES,
       edgecolor="#1A1A1A", linewidth=LINE_WIDTH, hatch="////",
       alpha=0.55, zorder=3)

for xi, v in zip(x, ACCURACY):
    ax.text(xi, v + 1.5, f"{v:.2f}", ha="center", va="bottom",
            fontsize=FONT_SIZE, color="#1A1A1A")
for xi, v in zip(x, REPLACE_RATE):
    ax.text(xi, -v * NEG_SCALE - 1.0, f"{v:.2f}", ha="center", va="top",
            fontsize=FONT_SIZE, color="#1A1A1A")

top = max(ACCURACY) * 1.32
bottom = -max(REPLACE_RATE) * NEG_SCALE * 1.55
ax.set_ylim(bottom, top)
ax.set_xlim(-0.6, len(CATEGORIES) - 0.4)

ax.set_yticks([0])
ax.set_yticklabels(["0"])

ax.axhline(0, color="#1A1A1A", linewidth=LINE_WIDTH, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(CATEGORIES, fontsize=FONT_SIZE)

zero_frac = -bottom / (top - bottom)
ax.text(-0.155, zero_frac + (1 - zero_frac) / 2, "Accuracy (%)",
        transform=ax.transAxes, rotation=90, va="center", ha="center",
        fontsize=FONT_SIZE)
ax.text(-0.155, zero_frac / 2, "Replace (%)",
        transform=ax.transAxes, rotation=90, va="center", ha="center",
        fontsize=FONT_SIZE)

ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(LINE_WIDTH)
    spine.set_color("#1A1A1A")
ax.set_box_aspect(1)

legend_handles = [
    Patch(facecolor="#8A8A8A", edgecolor="#1A1A1A", linewidth=LINE_WIDTH,
          label="AMC23 Avg@8"),
    Patch(facecolor="#8A8A8A", edgecolor="#1A1A1A", linewidth=LINE_WIDTH,
          hatch="////", alpha=0.55, label=r"$\eta(\tau_a)$"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=False,
          fontsize=FONT_SIZE, handlelength=1.5, handleheight=1.1,
          borderpad=0.3)

fig.tight_layout()

fig.savefig(f"{base}.png", dpi=300)
print(f"saved: {base}.png")
