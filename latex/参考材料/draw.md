# 图4 多模态表示融合与对齐实验记录

## 1. 实验目标

使用**训练完成模型中的真实隐藏表示**绘制图4，比较 NYC 数据集上三种模态 POI 表示在跨模态交互前后的分布变化，并补充高维空间中的定量统计结果。

## 2. 实验配置

- 数据集处理目录：`processed/NYC_coma_v8`
- 配置文件：`configs/nyc_v8_coma.yaml`
- 实际使用 checkpoint：`runs/fusion/NYC/cross_modal_modified/best.pt`
- 抽样 POI 数量：`1000`
- 随机种子：`42`
- 设备：`cuda:1`
- 融合方式：`cross_modal`

说明：原始 `runs/prior_conditioned/NYC/v8_coma/best.pt` 与当前代码中的 cross-modal 模块结构不兼容，因此本次实验改为使用与当前 `fusion.type: cross_modal` 结构匹配的真实 checkpoint：`runs/fusion/NYC/cross_modal_modified/best.pt`。

## 3. 表示定义

### 对齐前表示

来自三个独立模态编码器的真实 POI 级输出：

- `before_semantic.npy`
- `before_topology.npy`
- `before_spatial.npy`

### 对齐后表示

来自跨模态交互模块 `cross_modal_fusion` 输出、门控融合前的三模态表示：

- `after_semantic.npy`
- `after_topology.npy`
- `after_spatial.npy`

注意：这里没有把最终融合向量复制三份，而是保留了三种模态各自的交互后表示。

## 4. 输出文件

### Embedding 与统计

- `results/embeddings/nyc/poi_ids.npy`
- `results/embeddings/nyc/before_semantic.npy`
- `results/embeddings/nyc/before_topology.npy`
- `results/embeddings/nyc/before_spatial.npy`
- `results/embeddings/nyc/after_semantic.npy`
- `results/embeddings/nyc/after_topology.npy`
- `results/embeddings/nyc/after_spatial.npy`
- `results/embeddings/nyc/alignment_statistics.json`

### 图片

- `figure/nyc_embedding_tsne.pdf`
- `figure/nyc_embedding_tsne.png`
- `figure/nyc_similarity_distribution.png`

## 5. 图像

![图4：NYC 多模态表示 t-SNE 可视化](figure/nyc_embedding_tsne.png)

![正负样本跨模态相似度分布图](figure/nyc_similarity_distribution.png)

建议图注：

> 图4 NYC 数据集上跨模态交互与一致性约束前后的多模态 POI 表示 t-SNE 可视化。

英文：

> Fig. 4. t-SNE visualization of multimodal POI representations before and after cross-modal interaction and alignment on NYC.

## 6. 定量统计结果

### 6.1 同 POI / 不同 POI 跨模态余弦相似度

| Stage | Pair | Positive mean ± std | Negative mean ± std | Alignment gap |
| --- | --- | ---: | ---: | ---: |
| Before | Semantic–Topology | 0.0104 ± 0.0934 | 0.0092 ± 0.0925 | 0.0012 |
| Before | Semantic–Spatial | -0.0071 ± 0.0841 | -0.0139 ± 0.0829 | 0.0068 |
| Before | Topology–Spatial | 0.1438 ± 0.0899 | 0.1285 ± 0.0929 | 0.0152 |
| After | Semantic–Topology | -0.0844 ± 0.0642 | -0.0870 ± 0.0632 | 0.0026 |
| After | Semantic–Spatial | -0.0116 ± 0.0558 | -0.0163 ± 0.0540 | 0.0047 |
| After | Topology–Spatial | -0.0388 ± 0.0580 | -0.0696 ± 0.0681 | 0.0308 |

### 6.2 表示方差（用于检查是否坍缩）

| Stage | Semantic | Topology | Spatial |
| --- | ---: | ---: | ---: |
| Before | 0.9978 | 0.0297 | 0.4632 |
| After | 0.4325 | 0.1220 | 0.4798 |

### 6.3 正负样本相似度分布图的含义

本次新增的 `figure/nyc_similarity_distribution.png` 使用 **Semantic–Topology** 这组模态对做示例：

- **Positive pairs**：同一 POI 的语义表示与拓扑表示；
- **Negative pairs**：不同 POI 的语义表示与拓扑表示。

这张图的作用不是看二维投影形状，而是直接在**原始高维空间**里观察：

1. 正样本分布是否整体比负样本更靠右；
2. 对齐后正负样本是否拉开；
3. 是否出现“正负样本一起变高、几乎重叠”的坍缩风险。

如果模型对齐有效，理想情况通常是：

- 正样本分布右移；
- 负样本分布不要同步大幅右移；
- 两者均值差或分布间隔扩大。

## 7. 结果解读

基于当前真实实验结果，更稳妥的表述应为：

1. **图4中的散点来自真实训练后模型的隐藏表示，而非随机二维点。**
2. 在当前 checkpoint 下，跨模态交互后并**没有表现出三种模态全面显著靠近**的现象，因此不宜夸大为“所有模态都完成了强对齐”。
3. 从高维统计看，`Topology–Spatial` 的 alignment gap 从 `0.0152` 提升到 `0.0308`，说明这两种模态的同 POI / 异 POI 区分有所增强。
4. `Semantic–Topology` 与 `Semantic–Spatial` 的 alignment gap 没有同步明显提升，说明当前模型的跨模态交互收益具有**模态对选择性**。
5. 从方差看，三种模态在对齐后都**没有塌缩到接近 0**，尤其 spatial 方差基本保持稳定，topology 方差反而增大，因此目前没有明显证据表明表示发生整体坍缩；但 semantic 方差从 `0.9978` 下降到 `0.4325`，说明语义表示分布被明显压缩，论文中应谨慎解释。
6. 从新增的正负样本分布图看，`Semantic–Topology` 在对齐后**正样本和负样本都整体左移**，说明该模态对的平均余弦相似度整体下降；但由于负样本下降幅度略大于正样本，gap 仍从 `0.0012` 小幅增加到 `0.0026`。这意味着当前交互模块对这组模态并不是简单“拉近”，而更像是在重排其表示结构。

## 7.1 为什么“交互前更重叠，交互后反而更分开”？

这是**可能出现且并不矛盾**的现象，主要有四个原因：

1. **t-SNE 只保留局部邻域，不保留全局几何关系。**
   左右两图是分别运行的 t-SNE，因此它们的坐标系不同。某一阶段在二维上“看起来更混”，并不等价于原始高维空间里真的更接近；同理，“看起来更分开”也不一定表示高维上更远。

2. **交互模块的目标不是把三种模态简单压成一团，而是强化有用的对应关系。**
   当前结果里最明显的是 `Topology–Spatial` 的 alignment gap 从 `0.0152` 提升到 `0.0308`。这说明交互后，**同一 POI 与不同 POI 的可分性**在这一模态对上增强了。换句话说，模型可能让“该对齐的更对齐、该区分的更区分”，在二维图上就可能表现成局部结构更清晰、甚至视觉上更分散。

3. **“交互前重叠”有时只是尺度和方差造成的投影假象。**
   例如对齐前 topology 的方差只有 `0.0297`，明显小于 semantic 的 `0.9978` 和 spatial 的 `0.4632`。某个模态如果本身在高维空间里变化范围很小，t-SNE 投影后容易显得挤在一起，进而和别的模态“看起来重叠”；这不一定代表真正完成了跨模态对齐。

4. **当前 checkpoint 的交互收益具有“选择性”，不是全模态一致增强。**
   从统计看：
   - `Semantic–Topology` gap：`0.0012 -> 0.0026`，提升很小；
   - `Semantic–Spatial` gap：`0.0068 -> 0.0047`，略有下降；
   - `Topology–Spatial` gap：`0.0152 -> 0.0308`，提升最明显。

   这说明当前模型并不是把三种模态统一拉近，而更像是**重新组织了模态结构**：某些模态对更有对应性，某些模态对仍保持差异。因此二维图可能呈现“更分开但更有结构”的效果。

因此，更准确的结论不是“交互失败了”，而是：

> 在当前真实 checkpoint 下，跨模态交互并未使三种模态在二维投影中整体混合得更强，但它改变了表示结构，并在部分模态对（尤其拓扑-空间）上提升了同一 POI 与不同 POI 的区分间隔。

如果你希望把这个问题说得更扎实，下一步最值得补的是：

- 再加一组**跨模态检索指标**（Recall@1/5/10, MRR）；
- 或者把 before/after 的**正负样本相似度分布直方图**也画出来；
- 或者固定同一批 POI，再试 2~3 个 `random_state` 检查 t-SNE 视觉趋势是否稳定。

## 8. 正文可用描述（基于真实结果，谨慎版本）

> 为进一步分析跨模态交互对表示空间的影响，本文从 NYC 数据集中随机选取 1000 个具有完整模态信息且在训练集中出现过的 POI，分别提取三种独立模态编码后的 POI 级表示，以及经过跨模态交互后的模态表示，并在每个阶段对三种模态联合进行 t-SNE 降维。图4展示了交互前后多模态表示在二维投影空间中的分布变化。结合高维余弦相似度统计可以发现，当前模型并未使三种模态在所有配对上都出现一致性增强，但拓扑-空间模态对的正负样本间隔有所扩大，表明跨模态交互在部分模态关系上增强了同一 POI 的对应性，同时整体表示方差未出现接近零的坍缩现象。

## 9. 复现实验命令

### 提取真实 embedding

```bash
/opt/conda/envs/py11/bin/python scripts/extract_multimodal_embeddings.py \
  --config configs/nyc_v8_coma.yaml \
  --checkpoint runs/fusion/NYC/cross_modal_modified/best.pt \
  --output_dir results/embeddings/nyc \
  --sample_size 1000 \
  --seed 42
```

### 绘制 t-SNE 与统计

```bash
/opt/conda/envs/py11/bin/python scripts/plot_embedding_tsne.py \
  --embedding_dir results/embeddings/nyc \
  --output figure/nyc_embedding_tsne.pdf \
  --png_output figure/nyc_embedding_tsne.png \
  --random_state 42
```

## 10. 图5 门控权重热力图实验记录

### 10.1 实验目标

基于**真实测试样本**与**训练完成后的真实门控网络输出**，生成 NYC 数据集图5所需的：

- 三模态在不同上下文场景下的绝对平均门控权重；
- 相对全局均值偏移热力图；
- 场景样本数与距离分位数阈值。

### 10.2 实验配置

- 数据集处理目录：`processed/NYC_coma_v8`
- 实际使用 checkpoint：`runs/fusion/NYC/cross_modal_modified/best.pt`
- 融合方式：`cross_modal`
- 测试集样本数：`1071`
- 训练集样本数（用于距离分位数阈值统计）：`114321`
- 设备：`cuda:1`

说明：`runs/prior_conditioned/NYC/v8_coma/best.pt` 不包含当前门控融合所需的真实 `adaptive_fusion_gate` 参数，因此图5必须使用与当前 cross-modal + adaptive gate 结构一致的真实 checkpoint：`runs/fusion/NYC/cross_modal_modified/best.pt`。

### 10.3 门控权重顺序说明

模型内部真实门控输出顺序为：

```text
[topology, semantic, spatial]
```

导出 CSV 时已重新映射为：

- `semantic_weight`
- `topology_weight`
- `spatial_weight`

因此后续统计表与热力图中的模态命名是正确对应的。

### 10.4 输出文件

#### 逐样本记录

- `results/gates/nyc_gate_records.csv`
- `results/gates/nyc_gate_records.summary.json`
- `results/gates/nyc_gate_records_train.csv`
- `results/gates/nyc_gate_records_train.summary.json`

#### 聚合统计

- `results/gates/nyc_distance_thresholds.json`
- `results/gates/aggregated/gate_mean.csv`
- `results/gates/aggregated/gate_std.csv`
- `results/gates/aggregated/gate_count.csv`
- `results/gates/aggregated/gate_diagnostics.json`
- `results/gates/gate_heatmap_relative.csv`

#### 图片

- `figure/nyc_gate_absolute_heatmap.pdf`
- `figure/nyc_gate_absolute_heatmap.png`
- `figure/nyc_gate_relative_heatmap.pdf`
- `figure/nyc_gate_relative_heatmap.png`

### 10.5 图像

![图5：NYC 不同上下文场景下三种模态的平均门控权重](figure/nyc_gate_absolute_heatmap.png)

![图5：NYC 不同上下文场景下门控权重相对全局均值偏移](figure/nyc_gate_relative_heatmap.png)

建议图注：

> 图5 NYC 数据集上不同上下文场景下三种模态的平均门控权重，以及相对于全局均值的偏移热力图。

英文：

> Fig. 5. Average modality gate weights and their deviations from global averages under different contextual scenarios on NYC.

### 10.6 距离阈值与样本数

训练集距离分位数阈值：

- 短距离阈值（25% 分位数）：`0.1599778064 km`
- 长距离阈值（75% 分位数）：`5.5059612597 km`

测试集场景样本数：

| 场景 | 样本数 |
| --- | ---: |
| 工作日白天 | 463 |
| 工作日夜间 | 412 |
| 周末 | 196 |
| 短距离转移 | 240 |
| 长距离转移 | 316 |

### 10.7 绝对平均门控权重

| Modality | 工作日白天 | 工作日夜间 | 周末 | 短距离转移 | 长距离转移 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic | 0.044777 | 0.046704 | 0.043002 | 0.053358 | 0.042380 |
| Topology | 0.580712 | 0.580374 | 0.612620 | 0.524836 | 0.646149 |
| Spatial | 0.374510 | 0.372922 | 0.344378 | 0.421806 | 0.311471 |

检查结果：各列三模态权重和分别为：

- 工作日白天：`1.000000000`
- 工作日夜间：`1.000000003`
- 周末：`0.999999997`
- 短距离转移：`0.999999998`
- 长距离转移：`1.000000000`

说明门控导出、顺序映射与场景聚合均一致，没有出现明显维度错误或缺失值问题。

### 10.8 相对全局均值偏移

全局门控均值：

- Semantic：`0.045194`
- Topology：`0.586421`
- Spatial：`0.368385`

相对全局均值偏移：

| Modality | 工作日白天 | 工作日夜间 | 周末 | 短距离转移 | 长距离转移 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic | -0.000416 | 0.001511 | -0.002192 | 0.008164 | -0.002813 |
| Topology | -0.005709 | -0.006047 | 0.026199 | -0.061586 | 0.059727 |
| Spatial | 0.006126 | 0.004537 | -0.024007 | 0.053421 | -0.056914 |

### 10.9 结果解读

基于当前真实实验结果，更稳妥的表述应为：

1. **门控模块并非固定融合。** 三种门控权重在测试集上的标准差分别为：
   - semantic：`0.0359`
   - topology：`0.1632`
   - spatial：`0.1603`

   说明模型在不同样本上确实发生了显著的权重调整，而不是近似常数融合。

2. **Topology 是当前 checkpoint 的主导模态。** 在所有五个场景下，拓扑权重均为三者中最高，且全局平均达到 `0.5864`。

3. **短距离场景下空间模态更强、拓扑模态更弱。**
   - Spatial 在短距离场景相对全局均值提升 `+0.0534`；
   - Topology 在短距离场景相对全局均值下降 `-0.0616`。

   这说明模型在近距离转移时更依赖空间邻近性信息。

4. **长距离场景下拓扑模态更强、空间模态更弱。**
   - Topology 在长距离场景相对全局均值提升 `+0.0597`；
   - Spatial 在长距离场景下降 `-0.0569`。

   这说明当转移跨度较大时，模型更倾向于依赖拓扑迁移结构，而非几何邻近性。

5. **周末场景下拓扑权重相对增强。**
   周末 Topology 较全局均值提升 `+0.0262`，而 Spatial 下降 `-0.0240`，表明在周末活动模式下，模型仍更偏向利用结构性迁移规律。

6. **Semantic 权重整体较小但存在轻微场景波动。**
   语义模态在当前 checkpoint 中全局平均仅 `0.0452`，说明其贡献较弱；但在短距离场景下仍有一定增强（`+0.0082`）。因此正文不宜夸大“语义主导”结论。

### 10.10 正文可用描述（基于真实结果，谨慎版本）

> 为分析上下文感知门控模块的动态融合行为，本文在 NYC 测试集上记录了每个样本对应的语义、拓扑和空间模态门控权重，并依据本地时间与相邻签到距离统计不同场景下的平均权重。其中，短距离与长距离阈值分别由训练集转移距离的第 25 与第 75 百分位数确定。实验结果表明，门控权重并非固定常数，而会随上下文条件发生变化。具体而言，短距离场景下空间模态权重相对增强，而长距离场景下拓扑模态权重明显提升，说明模型能够根据移动上下文对不同信息源的贡献进行自适应调整。

### 10.11 复现实验命令

#### 导出测试集真实门控权重

```bash
/opt/conda/envs/py11/bin/python scripts/export_gate_weights.py \
  --checkpoint runs/fusion/NYC/cross_modal_modified/best.pt \
  --split test \
  --output results/gates/nyc_gate_records.csv
```

#### 导出训练集记录（用于距离阈值）

```bash
/opt/conda/envs/py11/bin/python scripts/export_gate_weights.py \
  --checkpoint runs/fusion/NYC/cross_modal_modified/best.pt \
  --split train \
  --output results/gates/nyc_gate_records_train.csv
```

#### 聚合图5统计值

```bash
/opt/conda/envs/py11/bin/python scripts/aggregate_gate_heatmap.py \
  --input results/gates/nyc_gate_records.csv \
  --train_input results/gates/nyc_gate_records_train.csv \
  --output_dir results/gates/aggregated \
  --thresholds_out results/gates/nyc_distance_thresholds.json \
  --relative_out results/gates/gate_heatmap_relative.csv
```

#### 绘制热力图

```bash
/opt/conda/envs/py11/bin/python scripts/plot_gate_heatmap.py \
  --absolute results/gates/aggregated/gate_mean.csv \
  --relative results/gates/gate_heatmap_relative.csv \
  --output_dir figure
```
