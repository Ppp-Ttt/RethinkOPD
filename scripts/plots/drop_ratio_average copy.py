import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

# ==== 配置区: 修改数据与样式 ====
# DROP 比例 (%) -> 各数据集 (Avg@8, Pass@8)
DATA = {
    0: {"AIME24": (10.83, 23.33), "AIME25": (9.17, 23.33),
        "AMC23": (41.88, 70.00), "MATH500": (66.60, 88.80),
        "Average": (32.12, 51.37)},
    10: {"AIME24": (11.67, 23.33), "AIME25": (5.83, 20.00),
         "AMC23": (44.38, 72.50), "MATH500": (70.05, 88.00),
         "Average": (32.98, 50.95)},
    15: {"AIME24": (11.67, 23.33), "AIME25": (7.08, 20.00),
         "AMC23": (46.56, 77.50), "MATH500": (70.10, 89.20),
         "Average": (33.85, 52.50)},
    20: {"AIME24": (14.17, 26.67), "AIME25": (11.67, 30.00),
         "AMC23": (46.56, 75.00), "MATH500": (70.15, 88.40),
         "Average": (35.64, 55.02)},
    25: {"AIME24": (11.25, 23.33), "AIME25": (8.75, 26.67),
         "AMC23": (47.50, 70.00), "MATH500": (71.63, 88.20),
         "Average": (34.78, 52.05)},
    30: {"AIME24": (10.42, 26.67), "AIME25": (8.75, 26.67),
         "AMC23": (50.00, 80.00), "MATH500": (70.78, 89.20),
         "Average": (34.99, 55.64)},
    50: {"AIME24": (10.83, 23.33), "AIME25": (6.67, 20.00),
         "AMC23": (45.63, 77.50), "MATH500": (70.88, 87.20),
         "Average": (33.50, 52.01)},
    75: {"AIME24": (10.00, 26.67), "AIME25": (8.30, 20.00),
         "AMC23": (41.56, 70.00), "MATH500": (70.08, 88.00),
         "Average": (32.49, 51.17)},
}

DATASET = "Average"
COLORS = {"Avg@8": "#4198ac", "Pass@8": "#ea9e58"}

FONT_SIZE = 28
LINE_WIDTH = 4
BORDER_WIDTH = 2             # 图像四周边框粗细 (独立于 LINE_WIDTH)
TICK_WIDTH = 2               # 刻度小横线粗细 (独立于 LINE_WIDTH)
MARKER_SIZE = 22
HIGHLIGHT_RATIO = 20            # 需要强调的 drop ratio
HIGHLIGHT_COLOR = "#dceaf5"     # 强调区浅色底纹
HIGHLIGHT_HALF_WIDTH = 0.4     # 阴影半宽 (以类别间距为单位)
OUTPUT_NAME = "drop_ratio_average.png"
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

ratios = sorted(DATA)
positions = range(len(ratios))
avg8 = [DATA[r][DATASET][0] for r in ratios]
pass8 = [DATA[r][DATASET][1] for r in ratios]

fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax2 = ax.twinx()

hi = ratios.index(HIGHLIGHT_RATIO)
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

ax.set_xlabel("Drop Ratio (%)", fontsize=FONT_SIZE)
ax.set_xticks(list(positions))
ax.set_xticklabels([f"{r}" for r in ratios])
ax.tick_params(axis="x", labelsize=FONT_SIZE)
ax.set_xlim(-0.6, len(ratios) - 0.4)
ax.tick_params(axis="x", labelsize=FONT_SIZE - 4)

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

_handles, _labels = ax.get_legend_handles_labels()
_handles2, _labels2 = ax2.get_legend_handles_labels()
ax.legend(_handles + _handles2, _labels + _labels2,
          fontsize=FONT_SIZE-4, loc="upper left", frameon=False)

fig.tight_layout()
fig.savefig(out_path, dpi=300)
print(f"saved: {out_path}")
