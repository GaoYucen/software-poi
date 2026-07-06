"""Generate the framework architecture diagram for the paper (Figure 1)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import platform

# Use Chinese-capable font on macOS
if platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
elif platform.system() == 'Linux':
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# Colors
c_modality = '#D6EAF8'    # light blue
c_align = '#D5F5E3'       # light green
c_fusion = '#FDEBD0'      # light orange
c_gpt = '#E8DAEF'         # light purple
c_output = '#FADBD8'      # light red
c_arrow = '#566573'

def draw_box(ax, x, y, w, h, color, text, fontsize=9, text_color='black'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#2C3E50', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, color=c_arrow):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# === Row 1: Multi-modal Encoders ===
row1_y = 5.8
box_h = 0.7
box_w = 1.6
gap = 0.25
start_x = 0.5

encoders = [
    ('Topology\n(Transit+Node2Vec)', c_modality),
    ('Semantic\n(all-MiniLM+Category)', c_modality),
    ('Spatial\n(GPS Coordinate)', c_modality),
    ('Temporal\n(Hour+Day+Interval)', c_modality),
]

for i, (label, color) in enumerate(encoders):
    x = start_x + i * (box_w + gap)
    draw_box(ax, x, row1_y, box_w, box_h, color, label, fontsize=8)

# Label for row 1
ax.text(start_x + (box_w*4 + gap*3)/2, row1_y + box_h + 0.25, 'Multi-modal POI Encoding',
        fontsize=11, fontweight='bold', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

# === Row 2: SemAlign Cross-modal Alignment ===
row2_y = 4.2
align_w = 3.8
align_x = (10 - align_w) / 2
draw_box(ax, align_x, row2_y, align_w, 0.9, c_align,
         'SemAlign Cross-modal Alignment\n(Instance-level + Feature-level InfoNCE)', fontsize=9)
ax.text(align_x + align_w/2, row2_y + 1.15, 'Cross-modal Alignment',
        fontsize=11, fontweight='bold', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

# Arrows from encoders to alignment
for i in range(4):
    x = start_x + i * (box_w + gap) + box_w/2
    draw_arrow(ax, x, row1_y, x, row2_y + 0.9)

# === Row 3: ContextFusion ===
row3_y = 2.6
fusion_w = 3.8
fusion_x = (10 - fusion_w) / 2
draw_box(ax, fusion_x, row3_y, fusion_w, 0.9, c_fusion,
         'ContextFusion Adaptive Gating\n(Context-aware Fusion)', fontsize=9)
ax.text(fusion_x + fusion_w/2, row3_y + 1.15, 'Adaptive Multi-modal Fusion',
        fontsize=11, fontweight='bold', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

# Arrow from alignment to fusion
draw_arrow(ax, 5, row2_y, 5, row3_y + 0.9)

# === Row 4: GPT-2 Backbone ===
row4_y = 1.0
gpt_w = 3.8
gpt_x = (10 - gpt_w) / 2
draw_box(ax, gpt_x, row4_y, gpt_w, 0.9, c_gpt,
         'GPT-2 Backbone (Selective FT)\n(Freeze Attn/MLP, Update LN/Embed)', fontsize=9)
ax.text(gpt_x + gpt_w/2, row4_y + 1.15, 'LLM Sequential Modeling',
        fontsize=11, fontweight='bold', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

# Arrow from fusion to GPT
draw_arrow(ax, 5, row3_y, 5, row4_y + 0.9)

# === Row 5: Output ===
row5_y = 0.1
out_w = 2.5
out_x = (10 - out_w) / 2
draw_box(ax, out_x, row5_y, out_w, 0.6, c_output,
         'Next POI Prediction', fontsize=11, text_color='#C0392B')

# Arrow from GPT to output
draw_arrow(ax, 5, row4_y, 5, row5_y + 0.6)

# Add a legend
legend_elements = [
    mpatches.Patch(facecolor=c_modality, edgecolor='#2C3E50', label='Modality Encoder'),
    mpatches.Patch(facecolor=c_align, edgecolor='#2C3E50', label='Cross-modal Alignment'),
    mpatches.Patch(facecolor=c_fusion, edgecolor='#2C3E50', label='Adaptive Fusion'),
    mpatches.Patch(facecolor=c_gpt, edgecolor='#2C3E50', label='LLM Backbone'),
    mpatches.Patch(facecolor=c_output, edgecolor='#2C3E50', label='Prediction Output'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)

plt.tight_layout()
plt.savefig('latex/figure/framework.png', dpi=300, bbox_inches='tight')
plt.close()
print("Framework figure saved to latex/figure/framework.png")