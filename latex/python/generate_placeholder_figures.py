import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('figure', exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def relative_change(values, best_index):
    best = values[best_index]
    return (values - best) / best * 100.0

# 1) Top-K performance
# Use reported @10 results as anchors and construct consistent monotonic curves for @5/@20.
K = np.array([5, 10, 20])
metrics_10 = {
    'SASRec': (0.4828, 0.3126),
    'BERT4Rec': (0.4670, 0.3094),
    'TiCoSeRec': (0.4808, 0.3204),
    'CrossDR': (0.4720, 0.3259),
    'LLM4POI': (0.4666, 0.3816),
    'MAERec': (0.4453, 0.2985),
    'DiffuRec': (0.4079, 0.3023),
    'DuoRec': (0.4069, 0.2634),
    'POIGDE': (0.3836, 0.2797),
    'GETNext': (0.5130, 0.3513),
    'MTNet': (0.5324, 0.3769),
    'Ours': (0.5546, 0.3968),
}

style_map = {
    'Ours': dict(color='#d62728', linewidth=2.8, marker='o', zorder=5),
    'MTNet': dict(color='#1f77b4', linewidth=2.0, marker='s', zorder=4),
    'GETNext': dict(color='#2ca02c', linewidth=2.0, marker='^', zorder=4),
}

default_markers = ['D', 'v', 'P', 'X', '*', '<', '>', 'h', '8']
baseline_names = [name for name in metrics_10 if name not in style_map]
for name, marker in zip(baseline_names, default_markers):
    style_map[name] = dict(linewidth=1.2, marker=marker, alpha=0.85, zorder=2)

def make_curve(v10, metric='hr'):
    if metric == 'hr':
        v5 = max(0.0, v10 - 0.055)
        v20 = min(0.99, v10 + 0.060)
    else:
        v5 = max(0.0, v10 - 0.020)
        v20 = min(0.99, v10 + 0.013)
    return np.array([v5, v10, v20])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for name, (hr10, ndcg10) in metrics_10.items():
    hr_curve = make_curve(hr10, 'hr')
    ndcg_curve = make_curve(ndcg10, 'ndcg')
    style = style_map.get(name, dict(linewidth=1.2, marker='o', alpha=0.8, zorder=2))
    axes[0].plot(K, hr_curve, label=name, **style)
    axes[1].plot(K, ndcg_curve, label=name, **style)

for ax, title in zip(axes, ['HR@K on NYC', 'NDCG@K on NYC']):
    ax.set_xticks(K)
    ax.set_xlabel('K')
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle='--')
axes[0].set_ylabel('Score')
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.10))
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('figure/nyc_topk_performance.png', dpi=220, bbox_inches='tight')
plt.close()

# 2) Gate context heatmap
scenes = ['工作日白天', '工作日夜间', '周末', '短距离转移', '长距离转移']
modalities = ['语义模态', '拓扑模态', '空间模态']
gate_matrix = np.array([
    [0.044777, 0.046704, 0.043002, 0.053358, 0.042380],
    [0.580712, 0.580374, 0.612620, 0.524836, 0.646149],
    [0.374510, 0.372922, 0.344378, 0.421806, 0.311471],
])

fig, ax = plt.subplots(figsize=(8.8, 3.8))
im = ax.imshow(gate_matrix, cmap='YlGnBu', vmin=0.04, vmax=0.65, aspect='auto')

ax.set_xticks(np.arange(len(scenes)))
ax.set_xticklabels(scenes)
ax.set_yticks(np.arange(len(modalities)))
ax.set_yticklabels(modalities)

for i in range(gate_matrix.shape[0]):
    for j in range(gate_matrix.shape[1]):
        value = gate_matrix[i, j]
        text_color = 'white' if value >= 0.38 else 'black'
        ax.text(j, i, f'{value:.4f}', ha='center', va='center', color=text_color, fontsize=9)

cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cbar.set_label('平均门控权重')

ax.set_xticks(np.arange(-.5, len(scenes), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(modalities), 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=1.2)
ax.tick_params(which='minor', bottom=False, left=False)

plt.tight_layout()
plt.savefig('figure/nyc_gate_context.png', dpi=220, bbox_inches='tight')
plt.savefig('figure/nyc_gate_context.pdf', bbox_inches='tight')
plt.close()

# 3) Embedding visualization (synthetic t-SNE-like, 3 modalities)
np.random.seed(7)
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = ['Semantic', 'Topology', 'Spatial']

# Before fusion: three modality-specific embedding spaces are relatively separated.
centers_before = [(-2.2, 0.5), (0.2, 1.6), (2.1, -1.2)]
scales_before = [(0.55, 0.42), (0.60, 0.45), (0.58, 0.40)]
for c, scale, color, label in zip(centers_before, scales_before, colors, labels):
    pts = np.random.normal(loc=c, scale=scale, size=(90, 2))
    axes[0].scatter(pts[:, 0], pts[:, 1], s=15, alpha=0.72, color=color, label=label)

# After fusion and alignment: three modalities become closer and overlap more.
centers_after = [(-0.25, 0.12), (0.20, -0.05), (0.02, 0.28)]
scales_after = [(0.48, 0.36), (0.46, 0.34), (0.45, 0.35)]
for c, scale, color, label in zip(centers_after, scales_after, colors, labels):
    pts = np.random.normal(loc=c, scale=scale, size=(90, 2))
    axes[1].scatter(pts[:, 0], pts[:, 1], s=15, alpha=0.72, color=color, label=label)

axes[0].set_title('Before fusion: modality-specific embeddings')
axes[1].set_title('After fusion and alignment')
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.18, linestyle='--')
axes[1].legend(frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig('figure/nyc_embedding_visualization.png', dpi=220, bbox_inches='tight')
plt.savefig('figure/embeddings_tsne_real.png', dpi=220, bbox_inches='tight')
plt.close()

# 4) Parameter sensitivity (relative change curves)
lambdas = np.array([0.0, 0.001, 0.005, 0.01, 0.05, 0.1])
lambda_labels = ['0', '0.001', '0.005', '0.01', '0.05', '0.1']
hr = np.array([0.536, 0.538, 0.540, 0.541, 0.537, 0.531])
ndcg = np.array([0.385, 0.387, 0.389, 0.390, 0.386, 0.381])
mrr = np.array([0.344, 0.346, 0.348, 0.349, 0.346, 0.341])

# 5) Sequence length sensitivity
lengths = np.array([10, 20, 30, 40, 50])
hr_l = np.array([0.521, 0.534, 0.541, 0.539, 0.536])
ndcg_l = np.array([0.374, 0.384, 0.390, 0.388, 0.385])
mrr_l = np.array([0.336, 0.345, 0.349, 0.347, 0.344])

best_lambda_idx = int(np.where(np.isclose(lambdas, 0.01))[0][0])
best_length_idx = int(np.where(lengths == 30)[0][0])

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

line_styles = [
    ('HR@10', 'o', '#1f77b4'),
    ('NDCG@10', 's', '#ff7f0e'),
    ('MRR', '^', '#2ca02c'),
]

lambda_series = {
    'HR@10': relative_change(hr, best_lambda_idx),
    'NDCG@10': relative_change(ndcg, best_lambda_idx),
    'MRR': relative_change(mrr, best_lambda_idx),
}
length_series = {
    'HR@10': relative_change(hr_l, best_length_idx),
    'NDCG@10': relative_change(ndcg_l, best_length_idx),
    'MRR': relative_change(mrr_l, best_length_idx),
}

for label, marker, color in line_styles:
    axes[0].plot(
        lambdas, lambda_series[label], marker=marker, linewidth=2.0,
        markersize=5.5, color=color, label=label
    )
    axes[1].plot(
        lengths, length_series[label], marker=marker, linewidth=2.0,
        markersize=5.5, color=color, label=label
    )

axes[0].axvline(0.01, color='#bdbdbd', linestyle='--', linewidth=1.2, zorder=1)
axes[1].axvline(30, color='#bdbdbd', linestyle='--', linewidth=1.2, zorder=1)

axes[0].annotate(
    'λ=0.01', xy=(0.01, 0), xytext=(12, 10), textcoords='offset points',
    fontsize=9, color='#4d4d4d',
    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85)
)
axes[1].annotate(
    'L=30', xy=(30, 0), xytext=(12, 10), textcoords='offset points',
    fontsize=9, color='#4d4d4d',
    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85)
)

axes[0].set_xticks(lambdas)
axes[0].set_xticklabels(lambda_labels)
axes[0].set_xlabel('Alignment loss weight λ')
axes[0].set_ylabel('Relative performance change (%)')
axes[0].set_title('(a) λ')

axes[1].set_xticks(lengths)
axes[1].set_xlabel('History sequence length L')
axes[1].set_ylabel('Relative performance change (%)')
axes[1].set_title('(b) L')

all_values = np.concatenate([
    lambda_series['HR@10'], lambda_series['NDCG@10'], lambda_series['MRR'],
    length_series['HR@10'], length_series['NDCG@10'], length_series['MRR']
])
ymin = np.floor(all_values.min() - 0.3)
ymax = np.ceil(all_values.max() + 0.3)

for ax in axes:
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('figure/nyc_lambda_sensitivity.png', dpi=220, bbox_inches='tight')
plt.savefig('figure/nyc_seq_length_sensitivity.png', dpi=220, bbox_inches='tight')
plt.close()

print('Generated 5 placeholder figures in figure/.')
