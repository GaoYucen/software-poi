# Nofix NYC 第三轮 Epoch 模型的 t-SNE 与 Gate 热力图重算记录

## 1. 任务说明

- 目标：在 NYC 数据集上，使用 nofix 第三轮 epoch 模型重新计算并覆盖以下两张图对应数值：
  - `figure/nyc_embedding_tsne.pdf`
  - `figure/nyc_gate_absolute_heatmap.pdf`
- 使用模型：`runs/fusion/NYC/multimodal_no_fixed_prior_e3_resume/best.pt`
- 配置文件：`configs/nyc_v8_coma_multimodal_no_fixed_prior.yaml`
- 数据目录：`processed/NYC_coma_v8`
- 随机种子：`42`
- 设备：`cuda:1`

## 2. 本次生成/覆盖的输出

### 2.1 t-SNE 与相似度统计

- `results/embeddings/nyc_nofix_e3/poi_ids.npy`
- `results/embeddings/nyc_nofix_e3/before_semantic.npy`
- `results/embeddings/nyc_nofix_e3/before_topology.npy`
- `results/embeddings/nyc_nofix_e3/before_spatial.npy`
- `results/embeddings/nyc_nofix_e3/after_semantic.npy`
- `results/embeddings/nyc_nofix_e3/after_topology.npy`
- `results/embeddings/nyc_nofix_e3/after_spatial.npy`
- `results/embeddings/nyc_nofix_e3/alignment_statistics.json`

### 2.2 图像文件

- `figure/nyc_embedding_tsne.pdf`
- `figure/nyc_embedding_tsne.png`
- `figure/nyc_similarity_distribution.png`
- `figure/nyc_gate_absolute_heatmap.pdf`
- `figure/nyc_gate_absolute_heatmap.png`
- `figure/nyc_gate_relative_heatmap.pdf`
- `figure/nyc_gate_relative_heatmap.png`

### 2.3 Gate 原始记录与聚合结果

- `results/gates/nofix/nyc_e3_gate_records.csv`
- `results/gates/nofix/nyc_e3_gate_records.summary.json`
- `results/gates/nofix/nyc_e3_gate_records_train.csv`
- `results/gates/nofix/nyc_e3_gate_records_train.summary.json`
- `results/gates/nofix/aggregated_e3/gate_mean.csv`
- `results/gates/nofix/aggregated_e3/gate_std.csv`
- `results/gates/nofix/aggregated_e3/gate_count.csv`
- `results/gates/nofix/aggregated_e3/gate_diagnostics.json`
- `results/gates/nofix/nyc_e3_distance_thresholds.json`
- `results/gates/nofix/nyc_e3_gate_heatmap_relative.csv`

## 3. 复现命令

### 3.1 提取 nofix epoch=3 embedding

```bash
/opt/conda/envs/py11/bin/python scripts/extract_multimodal_embeddings.py \
  --config configs/nyc_v8_coma_multimodal_no_fixed_prior.yaml \
  --checkpoint runs/fusion/NYC/multimodal_no_fixed_prior_e3_resume/best.pt \
  --output_dir results/embeddings/nyc_nofix_e3 \
  --sample_size 1000 \
  --seed 42
```

### 3.2 重绘 t-SNE 图

```bash
/opt/conda/envs/py11/bin/python scripts/plot_embedding_tsne.py \
  --embedding_dir results/embeddings/nyc_nofix_e3 \
  --output figure/nyc_embedding_tsne.pdf \
  --png_output figure/nyc_embedding_tsne.png \
  --similarity_output figure/nyc_similarity_distribution.png \
  --random_state 42
```

### 3.3 导出 gate 权重

```bash
/opt/conda/envs/py11/bin/python scripts/export_gate_weights.py \
  --checkpoint runs/fusion/NYC/multimodal_no_fixed_prior_e3_resume/best.pt \
  --split test \
  --output results/gates/nofix/nyc_e3_gate_records.csv
```

```bash
/opt/conda/envs/py11/bin/python scripts/export_gate_weights.py \
  --checkpoint runs/fusion/NYC/multimodal_no_fixed_prior_e3_resume/best.pt \
  --split train \
  --output results/gates/nofix/nyc_e3_gate_records_train.csv
```

### 3.4 聚合并重绘 gate 热力图

```bash
/opt/conda/envs/py11/bin/python scripts/aggregate_gate_heatmap.py \
  --input results/gates/nofix/nyc_e3_gate_records.csv \
  --train_input results/gates/nofix/nyc_e3_gate_records_train.csv \
  --output_dir results/gates/nofix/aggregated_e3 \
  --thresholds_out results/gates/nofix/nyc_e3_distance_thresholds.json \
  --relative_out results/gates/nofix/nyc_e3_gate_heatmap_relative.csv
```

```bash
/opt/conda/envs/py11/bin/python scripts/plot_gate_heatmap.py \
  --absolute results/gates/nofix/aggregated_e3/gate_mean.csv \
  --relative results/gates/nofix/nyc_e3_gate_heatmap_relative.csv \
  --output_dir figure
```

## 4. t-SNE / 对齐统计结果

### 4.1 实验摘要

- 抽样 POI 数量：`1000`
- 融合方式：`cross_modal`
- embedding 输出目录：`results/embeddings/nyc_nofix_e3`

### 4.2 同 POI / 异 POI 跨模态余弦相似度

| Stage | Pair | Positive mean ± std | Negative mean ± std | Alignment gap |
| --- | --- | ---: | ---: | ---: |
| Before | Semantic–Topology | 0.0059 ± 0.0935 | 0.0040 ± 0.0926 | 0.0020 |
| Before | Semantic–Spatial | -0.0146 ± 0.0827 | -0.0209 ± 0.0818 | 0.0063 |
| Before | Topology–Spatial | 0.1386 ± 0.0857 | 0.1148 ± 0.0939 | 0.0238 |
| After | Semantic–Topology | -0.0725 ± 0.0768 | -0.0826 ± 0.0734 | 0.0101 |
| After | Semantic–Spatial | -0.1727 ± 0.0696 | -0.1759 ± 0.0720 | 0.0032 |
| After | Topology–Spatial | -0.2847 ± 0.0944 | -0.3321 ± 0.0890 | 0.0474 |

### 4.3 表示方差

| Stage | Semantic | Topology | Spatial |
| --- | ---: | ---: | ---: |
| Before | 0.9591 | 0.0524 | 0.4522 |
| After | 0.5308 | 0.4930 | 0.2439 |

### 4.4 简要解读

1. `Semantic–Topology` 的 alignment gap 从 `0.0020` 提升到 `0.0101`，说明这组模态在交互后区分同一 POI / 不同 POI 的能力增强。
2. `Topology–Spatial` 的 alignment gap 从 `0.0238` 提升到 `0.0474`，是提升最明显的一组。
3. `Semantic–Spatial` 的 gap 从 `0.0063` 降到 `0.0032`，说明 nofix epoch=3 下该模态对并未同步获得增强。
4. 交互后 topology 方差从 `0.0524` 增加到 `0.4930`，semantic 方差从 `0.9591` 降到 `0.5308`，spatial 方差从 `0.4522` 降到 `0.2439`；整体没有出现接近 0 的全面坍缩。

## 5. Gate 热力图统计结果

### 5.1 实验摘要

- test 样本数：`1071`
- train 样本数：`114321`
- 模型内部门控顺序：`[topology, semantic, spatial]`
- 导出列顺序：`semantic_weight`, `topology_weight`, `spatial_weight`

### 5.2 距离阈值

- 短距离阈值：`0.159978 km`
- 长距离阈值：`5.505961 km`

### 5.3 不同场景下三模态平均门控权重（对应 Figure 5 absolute heatmap）

| Modality | 工作日白天 | 工作日夜间 | 周末 | 短距离转移 | 长距离转移 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic | 0.0639 | 0.0741 | 0.0661 | 0.0786 | 0.0677 |
| Topology | 0.7732 | 0.7680 | 0.7942 | 0.7449 | 0.7907 |
| Spatial | 0.1629 | 0.1579 | 0.1396 | 0.1765 | 0.1416 |

### 5.4 不同场景下三模态门控权重标准差

| Modality | 工作日白天 | 工作日夜间 | 周末 | 短距离转移 | 长距离转移 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic | 0.0468 | 0.0540 | 0.0508 | 0.0588 | 0.0488 |
| Topology | 0.0956 | 0.1001 | 0.0920 | 0.0972 | 0.0972 |
| Spatial | 0.0870 | 0.0914 | 0.0784 | 0.0947 | 0.0818 |

### 5.5 场景样本数

| Modality | 工作日白天 | 工作日夜间 | 周末 | 短距离转移 | 长距离转移 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic | 463 | 412 | 196 | 240 | 316 |
| Topology | 463 | 412 | 196 | 240 | 316 |
| Spatial | 463 | 412 | 196 | 240 | 316 |

### 5.6 全局均值与整体波动

- Global mean:
  - Semantic: `0.068256`
  - Topology: `0.775036`
  - Spatial: `0.156707`
- Global std:
  - Semantic: `0.050596`
  - Topology: `0.097070`
  - Spatial: `0.087543`

### 5.7 简要解读

1. nofix epoch=3 模型下，**Topology gate 明显占主导**，所有场景中都保持在 `0.7449 ~ 0.7942`。
2. **短距离转移** 场景下 spatial 权重最高（`0.1765`），同时 topology 权重最低（`0.7449`），说明近距离迁移时空间模态更容易被激活。
3. **长距离转移** 场景下 topology 权重回升到 `0.7907`，spatial 降到 `0.1416`，说明远距离迁移更多依赖拓扑结构而非纯空间邻近。
4. semantic 权重整体较低（全局均值 `0.0683`），但在**工作日夜间**和**短距离转移**场景略高，分别达到 `0.0741` 和 `0.0786`。

## 6. 备注

- `figure/nyc_embedding_tsne.pdf` 与 `figure/nyc_gate_absolute_heatmap.pdf` 已按本次 nofix epoch=3 数值重新生成。
- `plot_gate_heatmap.py` 执行时出现了 Matplotlib 中文字形缺失警告（DejaVu Sans 不含相关 CJK glyph），但**不影响数值统计与文件生成**。
