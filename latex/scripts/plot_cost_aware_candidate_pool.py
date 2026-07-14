import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei',
    'SimHei', 'Arial Unicode MS', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 仅保留三种主要策略：
# CACR：基于代价感知规则，自适应决定是否执行高代价检索。
# 固定：固定全量访问，即所有查询都执行高代价检索，高代价调用率固定为 100%。
# 随机：以随机跳过高代价检索作为对照。
strategies = ['count_entropy', 'full', 'random_skip']
display_names = ['CACR', '固定', '随机']
short_names = ['CACR', '固定', '随机']

candidate_recall = np.array([0.695612, 0.700280, 0.692810])
avg_latency_ms = np.array([2.927849, 3.472701, 3.205132])
expensive_call_rate = np.array([0.682540, 1.000000, 0.698413])
qps = np.array([341.547671, 287.960295, 311.999592])

hr10 = np.array([0.535948, 0.534080, 0.531279])
ndcg10 = np.array([0.385060, 0.384886, 0.377313])

base_idx = 0
delta_hr10 = hr10 - hr10[base_idx]
delta_ndcg10 = ndcg10 - ndcg10[base_idx]

colors = {
    'full': '#4C78A8',
    'count_entropy': '#54A24B',
    'random_skip': '#B279A2',
}


def add_outline(text_obj):
    text_obj.set_path_effects([
        pe.withStroke(linewidth=2.6, foreground='white'),
        pe.Normal(),
    ])


fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.15), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.05, wspace=0.08, hspace=0.05)

# (a) recall-latency trade-off
ax = axes[0]
label_offsets = {
    'count_entropy': (-2, 14),
    'full': (-18, 13),
    'random_skip': (16, -14),
}
for i, strategy in enumerate(strategies):
    edge = 'black' if strategy == 'count_entropy' else 'white'
    lw = 1.3 if strategy == 'count_entropy' else 0.9
    ax.scatter(
        avg_latency_ms[i], candidate_recall[i],
        s=125, color=colors[strategy], edgecolor=edge, linewidth=lw, zorder=3
    )
    dx, dy = label_offsets[strategy]
    label = ax.annotate(
        display_names[i],
        xy=(avg_latency_ms[i], candidate_recall[i]),
        xytext=(dx, dy),
        textcoords='offset points', fontsize=8.4,
        ha='center', va='center', zorder=5,
        annotation_clip=False,
    )
    add_outline(label)

ax.set_xlabel('平均检索延迟 / ms')
ax.set_ylabel('候选召回率 Recall@128')
ax.set_xlim(2.82, 3.58)
ax.set_ylim(0.6915, 0.7021)
ax.grid(alpha=0.22, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('(a) 候选召回与检索延迟折中', y=-0.18, fontsize=10.5)

# (b) ranking quality preservation
ax = axes[1]
x = np.arange(len(strategies))
width = 0.30
bars1 = ax.bar(x - width / 2, delta_hr10, width=width,
               color='#4C78A8', label=r'$\Delta$HR@10', zorder=3)
bars2 = ax.bar(x + width / 2, delta_ndcg10, width=width,
               color='#E45756', label=r'$\Delta$NDCG@10', zorder=3)
ax.axhline(0, color='0.45', linewidth=1.0)
ax.set_xticks(x, short_names, fontsize=8.6)
ax.set_xlabel('不同策略')
ax.set_ylabel('相对 CACR 的性能变化')
ax.set_ylim(-0.0105, 0.0050)
ax.grid(axis='y', alpha=0.22, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bars in (bars1, bars2):
    for rect in bars:
        h = rect.get_height()
        # CACR 的差值为 0，两组文字容易重叠，因此只在零基线处标一次。
        if abs(h) < 1e-12:
            continue
        offset = 0.00042 if h >= 0 else -0.00042
        text = ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + offset,
            f'{h:+.4f}',
            ha='center',
            va='bottom' if h >= 0 else 'top',
            fontsize=7.7,
            clip_on=False,
        )
        add_outline(text)

zero_text = ax.text(x[0], 0.00035, '基准（0）', ha='center', va='bottom', fontsize=7.7)
add_outline(zero_text)
ax.legend(frameon=False, loc='lower left', fontsize=8.2, ncol=1)
ax.set_title('(b) 最终推荐性能保持情况', y=-0.18, fontsize=10.5)

# (c) expensive access and throughput
ax = axes[2]
exp_pct = expensive_call_rate * 100
x = np.arange(len(strategies))

bars1 = ax.bar(x - width / 2, exp_pct, width=width,
               color='#9C755F', label='高代价调用率（%）', zorder=3)
ax.set_xticks(x, short_names, fontsize=8.6)
ax.set_xlabel('不同策略')
ax.set_ylabel('高代价调用率（%）', color='#9C755F')
ax.tick_params(axis='y', labelcolor='#9C755F')
ax.set_ylim(0, max(exp_pct.max(), 100) + 15)
ax.grid(axis='y', alpha=0.22, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax2 = ax.twinx()
bars2 = ax2.bar(x + width / 2, qps, width=width,
                color='#54A24B', label='QPS', zorder=3)
ax2.set_ylabel('QPS', color='#54A24B')
ax2.tick_params(axis='y', labelcolor='#54A24B')
ax2.spines['top'].set_visible(False)
ax2.set_ylim(0, qps.max() + 60)

for rect in bars1:
    h = rect.get_height()
    text = ax.text(
        rect.get_x() + rect.get_width() / 2,
        h + 2.0,
        f'{h:.1f}',
        ha='center', va='bottom', fontsize=7.7,
        clip_on=False,
    )
    add_outline(text)

for rect in bars2:
    h = rect.get_height()
    text = ax2.text(
            rect.get_x() + rect.get_width() / 2,
        h + 6.0,
        f'{h:.1f}',
        ha='center', va='bottom', fontsize=7.7,
        clip_on=False,
    )
    add_outline(text)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2,
          frameon=False, loc='upper right', bbox_to_anchor=(0.98, 0.98),
          fontsize=8.0, ncol=1)
ax.set_title('(c) 高代价访问削减与吞吐提升', y=-0.18, fontsize=10.5)

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figure')
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, 'nyc_cost_aware_candidate_pool.png')
pdf_path = os.path.join(out_dir, 'nyc_cost_aware_candidate_pool.pdf')
plt.savefig(png_path, dpi=320, bbox_inches='tight', pad_inches=0.08)
plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.08)
plt.close()

print(png_path)
print(pdf_path)
