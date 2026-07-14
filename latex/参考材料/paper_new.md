# v8 CoMa 配置模型实现与实验结果梳理

## 1. 摘要

本文档梳理当前 **v8 CoMa 配置**下的模型实现、实验协议与实验效果，重点使用 `CoMa_res.md` 中的最新结果，不再混入旧版 v8 标准协议结果。

当前实验的核心目标不是复现旧的 sliding-window 全量验证协议，而是让现有 Prior-Conditioned POI 推荐模型尽量向 **CoMaPOI** 的评测设定靠拢：

- 使用 recent-trajectory 风格测试；
- 过滤低频 POI 和低活跃用户；
- 将历史窗口长度扩展到 30；
- 但仍保留 `closed_world` 候选协议，因此 cold target 依旧无法召回。

在这一设置下：

- **NYC CoMa-style v8** 达到 **HR@10 = 0.5546 / NDCG@10 = 0.3968 / MRR = 0.3527**；
- **TKY CoMa-style v8** 达到 **HR@10 = 0.5661 / NDCG@10 = 0.3755 / MRR = 0.3246**。

其中，NYC 结果已经优于若干经典序列推荐 baseline，但仍落后于 GETNext、MTNet 和 CoMaPOI。本阶段最重要的结论是：**协议变化本身会显著抬升指标，而模型主体仍然是一个轻量级的 prior-conditioned reranker，不应将当前结果直接表述为完全复现 CoMaPOI。**

---

## 2. CoMa 风格实验协议

### 2.1 配置文件

本次主要使用以下两个配置：

- `configs/nyc_v8_coma.yaml`
- `configs/tky_v8_coma.yaml`

两者结构基本一致，核心字段如下：

```yaml
max_seq_len: 30
min_user_checkins: 30
min_poi_checkins: 10
split_protocol: coma_recent_trajectory
current_trajectory_len: 30
candidate_protocol: closed_world
epochs: 1
dynamic_candidates:
  num_candidates: 256
```

### 2.2 与普通 v8 协议的关键差异

与旧版 NYC v8 相比，CoMa 风格协议有四点本质变化：

1. **低频 POI 过滤**：先过滤低频 POI，再过滤用户，减轻长尾压力；
2. **低活跃用户过滤**：只保留签到较长的用户，减少稀疏序列；
3. **recent-trajectory 测试协议**：每位用户只构造一次最近轨迹测试样本，而非大量 sliding-window prefix 样本；
4. **历史窗口从 10 提升到 30**：测试时模型能看到更长的近期上下文。

因此，这组实验结果更适合解读为：**在一个更接近 CoMaPOI 的评测协议下，当前 v8 模型的上限能达到什么程度。**

### 2.3 仍保留的严格限制

虽然协议向 CoMaPOI 靠拢，但实现上仍保留：

- `candidate_protocol=closed_world`；
- 候选 POI 只能来自训练阶段见过的 POI；
- cold target 在评估中保留在分母中，但命中始终为 0。

因此，本实现相比 CoMaPOI 原始设定依然更严格，特别是在 cold target 处理上。

---

## 3. 模型实现梳理

当前主模型位于 `poi_rec/models/poi_model.py`，类名为 `PriorConditionedPOIModel`。整体结构可以概括为：

> **三模态 POI 编码 + 长期用户先验 + Transformer 序列编码 + 多模态动态候选召回 + 检索/语义联合打分**

### 3.1 三模态 POI 编码

三种 POI 编码器定义在 `poi_rec/models/encoders.py`：

1. **TopologyEncoder**
   - 输入：`transition_features`、`node2vec_embeddings`、`topology_available`
   - 输出：拓扑模态表示
   - 作用：利用 POI 转移图和 node2vec 图嵌入提供结构信息

2. **SemanticEncoder**
   - 输入：POI 类别 `poi_category` 与文本嵌入 `text_embeddings`
   - 输出：语义模态表示
   - 作用：融合类别与文本语义信息

3. **SpatialEncoder**
   - 输入：POI 坐标 `poi_coords`
   - 输出：空间模态表示
   - 作用：编码地理位置特征

三模态输出在 `poi_model.py` 中经过：

```python
self.modal = nn.Sequential(
    nn.Linear(self.hidden_dim * 3, self.hidden_dim),
    nn.GELU(),
    nn.LayerNorm(self.hidden_dim)
)
```

融合为统一的 POI token 表示。

### 3.2 长期用户先验

长期用户先验编码器位于 `poi_rec/models/prior_encoders.py` 中的 `LongTermPriorEncoder`。

它将以下用户级统计信息映射到一个 **profile token**：

- 类别分布；
- 时间分布；
- 空间中心与扩散；
- 历史 POI 频率加权表示。

这些统计量由 `PriorConditionedPOIModel._build_profiles()` 在初始化时预构建，并在前向时根据 `user_idx` 取出。这个 profile token 会被拼接到签到序列前端，作为用户长期偏好的压缩表示。

### 3.3 序列 token 构造

单个签到位置的 token 由三部分叠加构成：

1. 三模态 POI 融合表示；
2. 相对空间编码（当前位置相对最后一个 check-in 的坐标差）；
3. 时间嵌入（weekday × 24 + hour）。

对应实现：

```python
tokens = self.modal(torch.cat([topo, sem, spatial], -1)) \
    + self.relative_spatial(relative) \
    + time
```

之后将长期 profile token 拼接到前面，送入序列编码器。

### 3.4 序列编码器

当前 v8 CoMa 配置没有使用大型预训练 GPT，而是使用轻量级 TransformerEncoder：

```python
self.sequence_encoder = nn.TransformerEncoder(
    layer,
    int(seq.get("layers", 2)),
    norm=nn.LayerNorm(self.hidden_dim)
)
```

其特征为：

- 2 层 TransformerEncoder；
- hidden size = 128；
- 4 个注意力头；
- 使用 causal mask 保证自回归顺序。

编码后取最后一个有效位置的 hidden state 作为 `query_seed`。

### 3.5 候选打分与排序

候选表示由 `candidate_embeddings_for_ids()` 生成，本质上也是三模态编码后再经过 `candidate_projection`。

最终打分由三部分组成：

1. **query 与 candidate 的相似度得分**；
2. **retrieval_scores 检索分数偏置**；
3. **source_biases 候选来源偏置**。

核心形式为：

```python
scores = model_scores + self.retrieval_bias * retrieval_feature + source_bias
```

其中：

- `model_scores` 来自 query 与 candidate embedding 的归一化相似度；
- `retrieval_feature` 来自候选生成器的检索侧归一化分数；
- `source_bias` 来自候选来源位图 `source_flags` 的可学习偏置。

这说明当前模型并不是纯语义 reranker，而是 **检索分数 + 表征匹配 + 来源先验** 的联合排序器。

### 3.6 可选增强模块

代码中还支持若干可选增强：

- `target_time`
- `trajectory_neighbor`
- `category_auxiliary`
- `candidate_feature_mlp`
- `time_slot`
- `geo_cluster`

但就 **CoMa_res.md 当前使用的 base v8 CoMa 配置** 而言，主结论应聚焦于现有 base v8 CoMa 结果，不展开旧版 ablation 的优劣争论。

---

## 4. 多源动态候选生成实现

候选生成器位于 `poi_rec/data/candidates.py`，类名 `DynamicCandidateGenerator`。

### 4.1 基础 8 个候选源

当前基础 source 包括：

1. `history`
2. `transition`
3. `history_transition`
4. `user`
5. `spatial`
6. `temporal`
7. `popularity`
8. `category`

若启用 `trajectory_neighbor`，则会额外加入第 9 个 source。

### 4.2 候选构造过程

对每个样本，候选生成器会：

1. 从多个 source 收集候选 POI；
2. 用各 source 的预设权重累计分数；
3. 用 bit flag 记录候选来自哪些 source；
4. 按得分排序、截断到 `num_candidates=256`；
5. 返回：
   - `candidate_ids`
   - `candidate_mask`
   - `retrieval_scores`
   - `source_flags`

因此，候选生成不仅决定了召回上限，也直接向排序器提供了检索侧先验特征。

### 4.3 closed-world 约束

`_allowed()` 明确控制：

```python
return self.candidate_protocol != "closed_world" or bool(self.seen_mask[candidate])
```

这意味着：

- 若 POI 未在训练集中出现，则不会进入候选；
- 评估时这些 cold target 保留在分母里，但一定无法命中；
- 因而 `candidate_target_coverage` 是理解结果上限的关键指标。

---

## 5. 训练与评估链路

### 5.1 训练

训练逻辑在 `poi_rec/training/train.py`。

训练时的核心过程是：

1. DataLoader 读取样本；
2. `DynamicCandidateGenerator.build(batch)` 动态构造候选；
3. 模型前向输出候选分数；
4. 若目标 POI 在候选集中，则对该样本计算交叉熵损失；
5. 若目标不在候选中，该样本不参与推荐损失。

因此，训练本质是 **候选集内分类**，并非 full-softmax over all POIs。

### 5.2 评估

评估逻辑同样在 `train.py` 的 `evaluate_model()` 中，并由 `poi_rec/training/evaluate.py` 调用。

评估时会：

1. 仍然动态构造候选；
2. 将候选分数 scatter 到全量 POI 维度的 score matrix；
3. 未进入候选集的 POI 统一赋为 `-1e9`；
4. 计算 overall / warm / cold 三组指标。

因此：

- **overall** 反映真实评估分母下的最终效果；
- **warm** 只看训练见过的目标；
- **cold** 在 closed-world 下通常为 0。

在 CoMa 风格结果中，这一点同样成立：指标升高不只是模型变强，也来自数据过滤和 recent-trajectory 协议降低了任务难度。

---

## 6. CoMa 风格实验结果

## 6.1 NYC CoMa-style v8

配置：`configs/nyc_v8_coma.yaml`

### 数据统计

- `num_users`: 1072
- `num_pois`: 5118
- `train_seen_pois`: 5094
- `train samples`: 114321
- `test samples`: 1071

### Test 指标

| Metric | Value |
| --- | ---: |
| MRR | **0.352678** |
| HR@5 | 0.491130 |
| NDCG@5 | 0.376180 |
| HR@10 | **0.554622** |
| NDCG@10 | **0.396800** |
| HR@20 | 0.605976 |
| NDCG@20 | 0.409829 |
| candidate_target_coverage | 0.760971 |
| candidate_avg_size | 250.458450 |

### Warm 指标

| Metric | Value |
| --- | ---: |
| warm_MRR | **0.360763** |
| warm_HR@10 | **0.567335** |
| warm_NDCG@10 | **0.405896** |
| warm_candidate_target_coverage | 0.778415 |

### 结果解读

NYC CoMa-style 指标显著高于旧标准协议下的常见表现，原因主要来自：

- 长尾 POI 被过滤；
- 稀疏用户被过滤；
- 每用户仅保留一次 recent-trajectory 测试；
- 上下文长度提升到 30。

因此，这个结果应被视为 **在 CoMa 风格协议下的 v8 能力上限展示**，而不是对标准协议结果的直接替代。

## 6.2 TKY CoMa-style v8

配置：`configs/tky_v8_coma.yaml`

### 数据统计

- `num_users`: 2284
- `num_pois`: 7861
- `train_seen_pois`: 7858
- `train samples`: 376448
- `test samples`: 2284

### Test 指标

| Metric | Value |
| --- | ---: |
| MRR | **0.324629** |
| HR@5 | 0.463660 |
| NDCG@5 | 0.342125 |
| HR@10 | **0.566112** |
| NDCG@10 | **0.375514** |
| HR@20 | 0.643170 |
| NDCG@20 | 0.395054 |
| candidate_target_coverage | 0.827058 |
| candidate_avg_size | 251.894483 |

### 结果解读

TKY 的 candidate coverage 高于 NYC（0.8271 vs 0.7610），说明在 CoMa 风格协议下，TKY 的候选可覆盖性更好；但 TKY 的 NDCG@10 仍低于 NYC，说明 TKY 上的排序问题本身仍有一定难度。

---

## 7. 与 baseline 的对比（NYC）

以下 baseline 结果来自 `paper.md` 中整理的 **CoMaPOI 论文 NYC 对比表**。为了与 CoMa 风格 NYC v8 保持同一阅读上下文，这里将我们的 **NYC CoMa-style v8 overall 指标** 放入同一表格中。

| 分组 | Method | Venue | Year | HR@10 (%) | NDCG@10 (%) | MRR (%) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **比我们差 ↓** | SASRec | ICDM | 2018 | 48.28 | 31.26 | 27.01 |
|  | TiCoSeRec | AAAI | 2023 | 48.08 | 32.04 | 27.54 |
|  | CrossDR | ICWS | 2023 | 47.20 | 32.59 | 28.66 |
|  | BERT4Rec | CIKM | 2019 | 46.70 | 30.94 | 27.11 |
|  | LLM4POI | SIGIR | 2024 | 46.66 | 38.16 | 35.90 |
|  | MAERec | SIGIR | 2023 | 44.53 | 29.85 | 26.10 |
|  | DiffuRec | ACM TOIS | 2023 | 40.79 | 30.23 | 27.74 |
|  | DuoRec | WSDM | 2022 | 40.69 | 26.34 | 22.76 |
|  | POIGDE | ASC | 2024 | 38.36 | 27.97 | 25.40 |
|  | GETNext | SIGIR | 2022 | 51.30 | 35.13 | 30.97 |
|  | MTNet | AAAI | 2024 | 53.24 | 37.69 | 33.40 |
| **→ 我们的方法** | **PriorCond v8 CoMa-style (ours)** | — | 2026 | **55.46** | **39.68** | **35.27** |
| **比我们好 ↑** | CoMaPOI | SIGIR | 2025 | 59.01 | 42.82 | 37.67 |

### 对比解读

1. 从数值上看，**NYC CoMa-style v8** 已经高于表中的多数 baseline，包括 GETNext 和 MTNet；
2. 当前表中仍高于我们的只有 **CoMaPOI**，说明轻量 prior-conditioned 架构距离专门的多智能体 LLM 框架仍有差距；
3. 也应注意，这种对比带有协议近似而非完全一致的前提，因为我们仍然保留了 `closed_world` 约束；
4. 因此，更稳妥的表述应是：**当前实现已经在 CoMa 风格设定下达到较强的 baseline 水平，但尚不能声称严格复现或超越 CoMaPOI。**

> 注：表中的我们的结果使用 NYC CoMa-style v8 的 overall 指标，即 HR@10=55.46、NDCG@10=39.68、MRR=35.27。

---

## 8. 目前结论

综合当前代码实现与 CoMa 风格实验结果，可以得到以下结论：

1. **模型主体是轻量级 prior-conditioned reranker**，并非超大语言模型框架；
2. 它的优势来自：
   - 三模态 POI 表征；
   - 长期用户先验；
   - 多模态动态候选召回；
   - 检索分数与表示匹配联合打分；
3. **CoMa 风格协议显著抬升了 NYC/TKY 指标**，说明评测协议对最终表现影响很大；
4. 当前结果足以支持“该方法在更友好的 CoMa-style setting 下具有较强竞争力”；
5. 但由于仍保留 `closed_world`、且并非完全同协议实现，**不应直接将本结果等同于 CoMaPOI 的原论文结果。**

---

## 9. 后续建议

如果后续继续推进，建议优先关注以下方向：

1. **继续提升 candidate coverage**
   - 尤其是在 NYC 上，coverage 仍未到 0.8 以上；
   - 更强的候选召回可能继续带来整体 HR/NDCG 提升。

2. **做更严格的同协议 baseline 复核**
   - 若要在论文中更强地声称优于某些 baseline，需要进一步确认协议一致性。

3. **分析 posterior ranking 与 candidate miss 的相对瓶颈**
   - TKY 的 coverage 已较高，后续改进未必主要来自召回；
   - NYC 则仍可能同时受召回和排序限制。

4. **如果要进一步对齐 CoMaPOI，应继续缩小协议差异**
   - 尤其是 cold target 处理与候选机制差异。

---

## 10. 涉及的主要文件

- 配置：
  - `configs/nyc_v8_coma.yaml`
  - `configs/tky_v8_coma.yaml`
- 模型：
  - `poi_rec/models/poi_model.py`
  - `poi_rec/models/encoders.py`
  - `poi_rec/models/prior_encoders.py`
- 候选生成：
  - `poi_rec/data/candidates.py`
- 训练与评估：
  - `poi_rec/training/train.py`
  - `poi_rec/training/evaluate.py`
- 实验记录：
  - `CoMa_res.md`
  - `paper.md`
