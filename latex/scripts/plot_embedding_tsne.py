#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC", "PingFang SC", "Heiti SC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制真实多模态 POI 嵌入的 t-SNE 图，并计算对齐统计信息。")
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--output", default="figure/nyc_embedding_tsne.pdf")
    parser.add_argument("--png_output", default="figure/nyc_embedding_tsne.png")
    parser.add_argument("--similarity_output", default="figure/nyc_similarity_distribution.png")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def run_joint_tsne(semantic: np.ndarray, topology: np.ndarray, spatial: np.ndarray, random_state: int) -> dict[str, np.ndarray]:
    x = np.concatenate([semantic, topology, spatial], axis=0)
    x = StandardScaler().fit_transform(x)
    perplexity = min(30, max(5, (len(x) - 1) // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", max_iter=1500, random_state=random_state)
    points = tsne.fit_transform(x)
    n = len(semantic)
    return {"semantic": points[:n], "topology": points[n:2 * n], "spatial": points[2 * n:3 * n]}


def cosine_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    pos = np.sum(a_n * b_n, axis=1)
    sim = a_n @ b_n.T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    neg = sim[mask]
    return {
        "positive_mean": float(pos.mean()),
        "positive_std": float(pos.std(ddof=1)) if len(pos) > 1 else 0.0,
        "negative_mean": float(neg.mean()),
        "negative_std": float(neg.std(ddof=1)) if len(neg) > 1 else 0.0,
        "alignment_gap": float(pos.mean() - neg.mean()),
    }


def cosine_distributions(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    pos = np.sum(a_n * b_n, axis=1)
    sim = a_n @ b_n.T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    neg = sim[mask]
    return pos, neg


def embedding_variance(x: np.ndarray) -> float:
    return float(np.var(x, axis=0).mean())


def plot_panel(ax, points: dict[str, np.ndarray], title: str) -> None:
    ax.scatter(points["semantic"][:, 0], points["semantic"][:, 1], s=10, alpha=0.65, label="语义模态")
    ax.scatter(points["topology"][:, 0], points["topology"][:, 1], s=10, alpha=0.65, label="拓扑模态")
    ax.scatter(points["spatial"][:, 0], points["spatial"][:, 1], s=10, alpha=0.65, label="空间模态")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.075, title, transform=ax.transAxes, ha="center", va="top")


def plot_similarity_panel(ax, pos: np.ndarray, neg: np.ndarray, title: str) -> None:
    bins = np.linspace(-1.0, 1.0, 60)
    ax.hist(neg, bins=bins, alpha=0.55, density=True, label="负样本对", color="#dd8452")
    ax.hist(pos, bins=bins, alpha=0.55, density=True, label="正样本对", color="#4c72b0")
    ax.axvline(float(pos.mean()), color="#4c72b0", linestyle="--", linewidth=1.2)
    ax.axvline(float(neg.mean()), color="#dd8452", linestyle="--", linewidth=1.2)
    ax.set_xlabel("余弦相似度")
    ax.set_ylabel("密度")
    ax.text(0.5, -0.145, title, transform=ax.transAxes, ha="center", va="top")


def main() -> None:
    args = parse_args()
    embedding_dir = Path(args.embedding_dir)
    before = {
        "semantic": np.load(embedding_dir / "before_semantic.npy"),
        "topology": np.load(embedding_dir / "before_topology.npy"),
        "spatial": np.load(embedding_dir / "before_spatial.npy"),
    }
    after = {
        "semantic": np.load(embedding_dir / "after_semantic.npy"),
        "topology": np.load(embedding_dir / "after_topology.npy"),
        "spatial": np.load(embedding_dir / "after_spatial.npy"),
    }

    before_points = run_joint_tsne(before["semantic"], before["topology"], before["spatial"], args.random_state)
    after_points = run_joint_tsne(after["semantic"], after["topology"], after["spatial"], args.random_state)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    plot_panel(axes[0], before_points, "(a) 跨模态交互前")
    plot_panel(axes[1], after_points, "(b) 交互与对齐后")
    handles, labels = axes[0].get_legend_handles_labels()
    pdf_path = Path(args.output)
    png_path = Path(args.png_output)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0.06, 1, 0.92))
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    sim_before_pos, sim_before_neg = cosine_distributions(before["semantic"], before["topology"])
    sim_after_pos, sim_after_neg = cosine_distributions(after["semantic"], after["topology"])
    sim_fig, sim_axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_similarity_panel(sim_axes[0], sim_before_pos, sim_before_neg, "(a) 对齐前的语义—拓扑相似度")
    plot_similarity_panel(sim_axes[1], sim_after_pos, sim_after_neg, "(b) 对齐后的语义—拓扑相似度")
    sim_handles, sim_labels = sim_axes[0].get_legend_handles_labels()
    sim_path = Path(args.similarity_output)
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    sim_fig.legend(sim_handles, sim_labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=(0, 0.1, 1, 0.9))
    plt.savefig(sim_path, dpi=300, bbox_inches="tight")
    plt.close()

    stats = {
        "before": {
            "semantic_topology": cosine_stats(before["semantic"], before["topology"]),
            "semantic_spatial": cosine_stats(before["semantic"], before["spatial"]),
            "topology_spatial": cosine_stats(before["topology"], before["spatial"]),
            "variance": {k: embedding_variance(v) for k, v in before.items()},
        },
        "after": {
            "semantic_topology": cosine_stats(after["semantic"], after["topology"]),
            "semantic_spatial": cosine_stats(after["semantic"], after["spatial"]),
            "topology_spatial": cosine_stats(after["topology"], after["spatial"]),
            "variance": {k: embedding_variance(v) for k, v in after.items()},
        },
        "figure_pdf": str(pdf_path),
        "figure_png": str(png_path),
        "similarity_figure_png": str(sim_path),
    }
    (embedding_dir / "alignment_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()