#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
]
plt.rcParams["axes.unicode_minus"] = False


SCENES = ["工作日白天", "工作日夜间", "周末", "短距离转移", "长距离转移"]
MODALITIES = ["语义模态", "拓扑模态", "空间模态"]

ABSOLUTE = np.array(
    [
        [0.0639, 0.0741, 0.0661, 0.0786, 0.0677],
        [0.7732, 0.7680, 0.7942, 0.7449, 0.7907],
        [0.1629, 0.1579, 0.1396, 0.1765, 0.1416],
    ],
    dtype=float,
)

GLOBAL_MEAN = np.array([[0.068256], [0.775036], [0.156707]], dtype=float)
RELATIVE_DELTA = (ABSOLUTE - GLOBAL_MEAN) * 100.0  # percentage points


def annotate(ax, data: np.ndarray, fmt: str, threshold: float | None = None) -> None:
    if threshold is None:
        threshold = float((data.max() + data.min()) / 2.0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value >= threshold else "black"
            ax.text(j, i, fmt.format(value), ha="center", va="center", color=color, fontsize=9)


def style_heatmap(ax, title: str) -> None:
    ax.set_xticks(np.arange(len(SCENES)))
    ax.set_xticklabels(SCENES, rotation=0)
    ax.set_yticks(np.arange(len(MODALITIES)))
    ax.set_yticklabels(MODALITIES)
    ax.set_xticks(np.arange(-0.5, len(SCENES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODALITIES), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(0.5, -0.14, title, transform=ax.transAxes, ha="center", va="top", fontsize=11)


def main() -> None:
    output_dir = Path("figure")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), gridspec_kw={"width_ratios": [1, 1]})

    im0 = axes[0].imshow(ABSOLUTE, cmap="YlGnBu", vmin=0.0, vmax=0.82, aspect="auto")
    style_heatmap(axes[0], "(a) 绝对平均门控权重")
    annotate(axes[0], ABSOLUTE, "{:.4f}", threshold=0.42)
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.set_label("平均门控权重")

    delta_max = float(np.abs(RELATIVE_DELTA).max())
    im1 = axes[1].imshow(RELATIVE_DELTA, cmap="RdBu_r", vmin=-delta_max, vmax=delta_max, aspect="auto")
    style_heatmap(axes[1], "(b) 相对全局均值的场景偏移（百分点）")
    for i in range(RELATIVE_DELTA.shape[0]):
        for j in range(RELATIVE_DELTA.shape[1]):
            value = RELATIVE_DELTA[i, j]
            color = "white" if abs(value) >= delta_max * 0.45 else "black"
            axes[1].text(j, i, f"{value:+.2f}", ha="center", va="center", color=color, fontsize=9)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cbar1.set_label("相对偏移（百分点）")

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(output_dir / "nyc_gate_context.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "nyc_gate_context.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()