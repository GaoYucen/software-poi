"""Generate the auditable architecture diagram used as Figure 1."""

import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


if platform.system() == "Darwin":
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
else:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 6.8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

colors = {
    "data": "#EAF2F8",
    "align": "#D5F5E3",
    "fusion": "#FDEBD0",
    "sequence": "#E8DAEF",
    "score": "#FADBD8",
    "prior": "#FCF3CF",
}


def box(x, y, width, height, text, color, fontsize=9):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.08", facecolor=color,
        edgecolor="#34495E", linewidth=1.35,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(x1, y1, x2, y2, style="-"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": "#566573", "lw": 1.6, "linestyle": style},
    )


# POI- and event-level encoders
box(0.3, 5.75, 2.15, 0.8, "拓扑模态\n转移统计 + Node2Vec", colors["data"])
box(2.7, 5.75, 2.15, 0.8, "语义模态\n文本编码 + 类别嵌入", colors["data"])
box(5.1, 5.75, 1.85, 0.8, "空间模态\n归一化坐标", colors["data"])
box(7.2, 5.75, 1.85, 0.8, "时间模态\n小时/星期/间隔", colors["data"])
box(9.3, 5.75, 2.35, 0.8, "用户上下文\nID嵌入 + 类别画像", colors["data"])
ax.text(6, 6.78, "多粒度轨迹信息编码", ha="center", fontsize=12, fontweight="bold")

# Only topology and semantics enter SemAlign
box(1.35, 4.25, 4.45, 0.9, "SemAlign 结构--语义对齐\n实例级 + 特征级对比目标", colors["align"], 10)
arrow(1.38, 5.75, 2.35, 5.15)
arrow(3.78, 5.75, 4.75, 5.15)

# ContextFusion consumes aligned views and the remaining context
box(3.25, 2.75, 5.5, 0.95, "ContextFusion 上下文自适应融合\n拓扑--语义门控 + 空间/时间/用户信息", colors["fusion"], 10)
arrow(3.58, 4.25, 5.25, 3.7)
arrow(6.02, 5.75, 6.0, 3.7)
arrow(8.12, 5.75, 7.0, 3.7)
arrow(10.48, 5.75, 7.95, 3.7)

box(3.25, 1.35, 5.5, 0.85, "预训练自回归Transformer\n128维输入投影 → GPT-2选择性微调 → 128维输出投影", colors["sequence"], 9.5)
arrow(6, 2.75, 6, 2.2)

box(3.25, 0.15, 5.5, 0.7, "查询--候选归一化匹配与Top-$K$排序", colors["score"], 10)
arrow(6, 1.35, 6, 0.85)

box(9.45, 0.1, 2.2, 0.8, "训练集统计先验\n混合排序扩展", colors["prior"], 9)
arrow(9.45, 0.5, 8.75, 0.5, "--")

ax.text(
    0.35, 0.18,
    "纯神经模型与统计先验扩展分开评价；所有行为统计仅由训练集构建",
    fontsize=8.5, color="#566573",
)

plt.tight_layout()
plt.savefig("latex/figure/framework.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Framework figure saved to latex/figure/framework.png")
