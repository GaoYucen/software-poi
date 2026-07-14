import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Arial Unicode MS', 'Microsoft YaHei',
    'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def normalized(values, base_index):
    values = np.asarray(values, dtype=float)
    return values / values[base_index]


def annotate_all_values(ax, xs, ys, abs_values, color, offsets):
    for x, y, value, (dx, dy) in zip(xs, ys, abs_values, offsets):
        text = ax.annotate(
            f'{value:.4f}',
            xy=(x, y),
            xytext=(dx, dy),
            textcoords='offset points',
            ha='center',
            va='center',
            fontsize=7.6,
            color=color,
            zorder=6,
        )
        text.set_path_effects([
            pe.withStroke(linewidth=2.8, foreground='white'),
            pe.Normal(),
        ])


out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figure_optimized_v4')
os.makedirs(out_dir, exist_ok=True)

# 真实实验结果
lambdas = np.array([0.005, 0.01, 0.05])
lambda_labels = ['0.005', '0.01', '0.05']
lambda_x = np.arange(len(lambdas))
lambda_hr10 = np.array([0.526533, 0.525677, 0.528340])
lambda_ndcg10 = np.array([0.383055, 0.382764, 0.384122])
lambda_mrr = np.array([0.344119, 0.344217, 0.344903])
lambda_base_idx = 1

lengths = np.array([10, 20, 30, 50])
length_labels = ['10', '20', '30', '50']
length_x = np.arange(len(lengths))
length_hr10 = np.array([0.517689, 0.524726, 0.525677, 0.527389])
length_ndcg10 = np.array([0.378107, 0.384219, 0.382764, 0.383734])
length_mrr = np.array([0.340493, 0.345981, 0.344217, 0.344609])
length_base_idx = 2

series_styles = [
    ('HR@10', '#1f77b4', 'o'),
    ('NDCG@10', '#d62728', 's'),
    ('MRR', '#2ca02c', '^'),
]

lambda_abs = {
    'HR@10': lambda_hr10,
    'NDCG@10': lambda_ndcg10,
    'MRR': lambda_mrr,
}
length_abs = {
    'HR@10': length_hr10,
    'NDCG@10': length_ndcg10,
    'MRR': length_mrr,
}
lambda_norm = {k: normalized(v, lambda_base_idx) for k, v in lambda_abs.items()}
length_norm = {k: normalized(v, length_base_idx) for k, v in length_abs.items()}

lambda_offsets = {
    'HR@10': [(0, 12), (0, 20), (0, 12)],
    'NDCG@10': [(0, -11), (0, 0), (0, -11)],
    'MRR': [(0, -12), (0, -20), (0, -12)],
}
length_offsets = {
    'HR@10': [(0, -12), (0, 12), (0, 20), (0, 12)],
    'NDCG@10': [(0, -12), (0, -12), (0, 0), (0, 5)],
    'MRR': [(0, 12), (0, 12), (0, -20), (0, -12)],
}

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.35))

for label, color, marker in series_styles:
    axes[0].plot(
        lambda_x, lambda_norm[label],
        color=color, marker=marker,
        linewidth=1.8, markersize=5.5,
        label=label, zorder=3,
    )
    axes[1].plot(
        length_x, length_norm[label],
        color=color, marker=marker,
        linewidth=1.8, markersize=5.5,
        label=label, zorder=3,
    )

    annotate_all_values(
        axes[0], lambda_x, lambda_norm[label],
        lambda_abs[label], color, lambda_offsets[label]
    )
    annotate_all_values(
        axes[1], length_x, length_norm[label],
        length_abs[label], color, length_offsets[label]
    )

# 默认参数提示
axes[0].axvline(lambda_base_idx, color='0.58', linestyle='--', linewidth=1.0, alpha=0.75, zorder=1)
axes[1].axvline(length_base_idx, color='0.58', linestyle='--', linewidth=1.0, alpha=0.75, zorder=1)
axes[0].text(lambda_base_idx, 0.975, '默认值', transform=axes[0].get_xaxis_transform(),
             ha='center', va='top', fontsize=8, color='0.38')
axes[1].text(length_base_idx, 0.975, '默认值', transform=axes[1].get_xaxis_transform(),
             ha='center', va='top', fontsize=8, color='0.38')

axes[0].set_xticks(lambda_x, lambda_labels)
axes[0].set_xlabel(r'对齐损失权重 $\lambda$', labelpad=5)
axes[0].set_ylabel('相对默认参数的归一化性能')

axes[1].set_xticks(length_x, length_labels)
axes[1].set_xlabel(r'历史序列长度 $L$', labelpad=5)
axes[1].set_ylabel('相对默认参数的归一化性能')

axes[0].text(0.5, -0.27, r'(a) $\lambda$ 敏感性', transform=axes[0].transAxes,
             ha='center', va='top', fontsize=10.5)
axes[1].text(0.5, -0.27, r'(b) $L$ 敏感性', transform=axes[1].transAxes,
             ha='center', va='top', fontsize=10.5)

lambda_values = np.concatenate(list(lambda_norm.values()))
length_values = np.concatenate(list(length_norm.values()))
axes[0].set_ylim(lambda_values.min() - 0.0023, lambda_values.max() + 0.0030)
axes[1].set_ylim(length_values.min() - 0.0042, length_values.max() + 0.0043)

for ax in axes:
    ax.axhline(1.0, color='0.72', linestyle=':', linewidth=1.0, zorder=1)
    ax.grid(axis='y', alpha=0.22, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.985),
    bbox_transform=fig.transFigure,
    handlelength=2.4,
    columnspacing=1.8,
)

# 保持两个子图等宽，并给上方图例、下方标题留足空间
fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.25, wspace=0.25)

png_path = os.path.join(out_dir, 'nyc_parameter_sensitivity_optimized_v4.png')
pdf_path = os.path.join(out_dir, 'nyc_parameter_sensitivity_optimized_v4.pdf')
plt.savefig(png_path, dpi=320)
plt.savefig(pdf_path)
plt.close()

print(png_path)
print(pdf_path)
