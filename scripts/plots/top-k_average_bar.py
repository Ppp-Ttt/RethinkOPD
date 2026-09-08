import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ==== 配置区: 修改数据与样式 ====
# top-k -> 各数据集 (Avg@8, Pass@8)
DATA = {
    4: {"AIME24": (14.17, 7.50), "AIME25": (26.67, 20.00),
        "AMC23": (44.69, 72.50), "MATH500": (70.30, 88.00),
        "Average": (34.17, 51.79)},
    8: {"AIME24": (11.67, 9.58), "AIME25": (23.33, 26.67),
        "AMC23": (46.25, 77.50), "MATH500": (70.40, 88.40),
        "Average": (34.48, 53.98)},
    16: {"AIME24": (14.17, 11.67), "AIME25": (26.67, 30.00),
         "AMC23": (46.56, 75.00), "MATH500": (70.15, 88.40),
         "Average": (35.64, 55.02)},
    32: {"AIME24": (14.17, 10.83), "AIME25": (26.67, 30.00),
         "AMC23": (46.25, 70.00), "MATH500": (70.53, 88.60),
         "Average": (35.45, 53.82)},
}

DATASET = "Average"
BAR_COLOR = "#4198ac"           # Avg@8 配色, 与折线版本一致

FONT_SIZE = 24
LINE_WIDTH = 4
BAR_WIDTH = 0.55
HIGHLIGHT_RATIO = 16            # 需要强调的 top-k
HIGHLIGHT_COLOR = "#dceaf5"     # 强调区浅色底纹
HIGHLIGHT_HALF_WIDTH = 0.42     # 阴影半宽 (略宽于柱体, 完整包住高亮柱)
OUTPUT_NAME = "top-k_average_bar.png"
# ================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT_SIZE,
    "axes.linewidth": LINE_WIDTH,
    "axes.edgecolor": "#1A1A1A",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": LINE_WIDTH,
    "ytick.major.width": LINE_WIDTH,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "savefig.bbox": "tight",
})

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)

ratios = sorted(DATA)
positions = np.arange(len(ratios))
avg8 = [DATA[r][DATASET][0] for r in ratios]

fig, ax = plt.subplots(figsize=(8.0, 6.0))

hi = ratios.index(HIGHLIGHT_RATIO)
ax.axvspan(hi - HIGHLIGHT_HALF_WIDTH, hi + HIGHLIGHT_HALF_WIDTH,
           color=HIGHLIGHT_COLOR, zorder=0)

ax.bar(positions, avg8, width=BAR_WIDTH, color=BAR_COLOR,
       edgecolor="#1A1A1A", linewidth=LINE_WIDTH, zorder=3, label="Avg@8")

for xi, v in zip(positions, avg8):
    ax.text(xi, v + 0.12, f"{v:.2f}", ha="center", va="bottom",
            fontsize=FONT_SIZE, color="#1A1A1A", zorder=4)

ax.set_xlabel("Top-k", fontsize=FONT_SIZE)
# ax.set_ylabel("Average Avg@8(%)", fontsize=FONT_SIZE)
ax.set_xticks(list(positions))
ax.set_xticklabels([f"{r}" for r in ratios])
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.set_xlim(-0.6, len(ratios) - 0.4)
ax.set_ylim(29, 38)
ax.set_yticks([29, 32, 35, 38])

ax.grid(False)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(LINE_WIDTH)
    spine.set_color("#1A1A1A")

# ax.legend(fontsize=FONT_SIZE, loc="upper left", frameon=False)

fig.tight_layout()
fig.savefig(out_path, dpi=300)
print(f"saved: {out_path}")
