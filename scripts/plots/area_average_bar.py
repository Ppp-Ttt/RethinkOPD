import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

# ==== 配置区: 修改数据与样式 ====
# area -> 各数据集 (Avg@8, Pass@8)
DATA = {
    "only_stu": {
        "Average": (34.15, 54.18)},
    "only_tch": {
        "Average": (34.25, 52.67)},
    "union": {
         "Average": (35.64, 55.02)},
}

DATASET = "Average"
XLABELS = {"only_stu": "Student", "only_tch": "Teacher", "union": "Union"}
COLORS = {"Avg@8": "#4198ac", "Pass@8": "#ea9e58"}

FONT_SIZE = 28
LINE_WIDTH = 4
BORDER_WIDTH = 2             # 图像四周边框粗细 (独立于 LINE_WIDTH)
TICK_WIDTH = 2               # 刻度小横线粗细 (独立于 LINE_WIDTH)
MARKER_SIZE = 22
HIGHLIGHT_KEY = "union"          # 需要强调的类别
HIGHLIGHT_COLOR = "#dceaf5"     # 强调区浅色底纹
HIGHLIGHT_HALF_WIDTH = 0.27     # 阴影半宽 (按视觉对齐 top-k_average)
OUTPUT_NAME = "area_average.png"
# ================================

# 四角星轮廓 (外顶点在上下左右, 内顶点收窄成星形)
_r_in = 0.50
_star4 = []
for i in range(4):
    ang = np.pi / 2 - i * np.pi / 2
    _star4.append((np.cos(ang), np.sin(ang)))
    ang -= np.pi / 4
    _star4.append((_r_in * np.cos(ang), _r_in * np.sin(ang)))
STAR4 = Path(_star4 + [_star4[0]], closed=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT_SIZE,
    "axes.linewidth": BORDER_WIDTH,
    "axes.edgecolor": "#1A1A1A",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "savefig.bbox": "tight",
})

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)

keys = list(DATA)
positions = range(len(keys))
avg8 = [DATA[k][DATASET][0] for k in keys]
pass8 = [DATA[k][DATASET][1] for k in keys]

fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax2 = ax.twinx()

hi = keys.index(HIGHLIGHT_KEY)
ax.axvspan(hi - HIGHLIGHT_HALF_WIDTH, hi + HIGHLIGHT_HALF_WIDTH,
           color=HIGHLIGHT_COLOR, zorder=0)

ax2.plot(positions, pass8, marker=STAR4, markersize=MARKER_SIZE,
         markerfacecolor="white", markeredgewidth=LINE_WIDTH,
         markeredgecolor=COLORS["Pass@8"],
         linewidth=LINE_WIDTH, color=COLORS["Pass@8"], label="Pass@8")
ax.plot(positions, avg8, marker=STAR4, markersize=MARKER_SIZE,
        markerfacecolor="white", markeredgewidth=LINE_WIDTH,
        markeredgecolor=COLORS["Avg@8"],
        linewidth=LINE_WIDTH, color=COLORS["Avg@8"], label="Avg@8")

LABEL_OFFSET = 1.0
for x, y in zip(positions, pass8):
    ax2.annotate(f"{y:.2f}", (x, y), xytext=(x, y + LABEL_OFFSET),
                 ha="center", va="bottom", fontsize=FONT_SIZE - 6,
                 color=COLORS["Pass@8"])
for x, y in zip(positions, avg8):
    ax.annotate(f"{y:.2f}", (x, y), xytext=(x, y - LABEL_OFFSET),
                ha="center", va="top", fontsize=FONT_SIZE - 6,
                color=COLORS["Avg@8"])

ax.set_xlabel("", fontsize=FONT_SIZE)
ax.set_xticks(list(positions))
ax.set_xticklabels([XLABELS[k] for k in keys])
ax.tick_params(axis="x", labelsize=FONT_SIZE - 4)
ax.set_xlim(-0.6, len(keys) - 0.4)

ax.set_ylim(29, 46)
ax.set_yticks([30, 35, 40, 45])
ax.tick_params(axis="y", labelsize=FONT_SIZE - 4, labelcolor=COLORS["Avg@8"],
               colors="#1A1A1A")

ax2.set_ylim(44, 61)
ax2.set_yticks([45, 50, 55, 60])
ax2.tick_params(axis="y", labelsize=FONT_SIZE - 4, labelcolor=COLORS["Pass@8"],
                colors="#1A1A1A")

ax.grid(False)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_linewidth(BORDER_WIDTH)
    s.set_color("#1A1A1A")
for s in ax2.spines.values():
    s.set_linewidth(BORDER_WIDTH)
    s.set_color("#1A1A1A")

# ax.legend(fontsize=FONT_SIZE, loc="center right", frameon=False)

fig.tight_layout()
fig.savefig(out_path, dpi=300)
print(f"saved: {out_path}")
