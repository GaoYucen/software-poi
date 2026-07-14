# CoMaPOI NYC 消融实验

## 1. 实验目标

针对 NYC CoMa 对齐版本，补充两个消融实验，用于拆分 **候选检索先验** 与 **神经 reranker** 的实际贡献：

- **Retrieval Only**：候选集合保持不变，仅按候选检索分数排序。
- **Neural Only**：候选集合保持不变，仅按模型神经分数排序。

对应解释为：

- **Retrieval Only** 用来衡量：如果去掉神经 reranker，仅保留统计/启发式候选检索先验，系统本身能达到什么水平。
- **Neural Only** 用来衡量：如果不使用检索先验加成，只在相同候选集上依赖神经模型排序，模型本身能达到什么水平。

---

## 2. 实验设置

### 数据与模型

- 配置文件：`configs/nyc_v8_coma.yaml`
- checkpoint：`runs/prior_conditioned/NYC/v8_coma/best.pt`
- 测试集：`test`
- 候选协议：`closed_world`
- 候选集合：与 baseline 完全一致，不重新构造候选，只改变排序分数来源。

### 评分模式

本次为评估脚本新增了 `--score-mode` 开关：

- `full`：使用原始融合分数 `scores`
- `retrieval_only`：只使用候选检索分数 `retrieval_scores`
- `neural_only`：只使用模型神经分数 `model_scores`

### 评估命令

```bash
/opt/conda/envs/py11/bin/python scripts/evaluate.py \
  --checkpoint runs/prior_conditioned/NYC/v8_coma/best.pt \
  --split test \
  --score-mode full

/opt/conda/envs/py11/bin/python scripts/evaluate.py \
  --checkpoint runs/prior_conditioned/NYC/v8_coma/best.pt \
  --split test \
  --score-mode retrieval_only

/opt/conda/envs/py11/bin/python scripts/evaluate.py \
  --checkpoint runs/prior_conditioned/NYC/v8_coma/best.pt \
  --split test \
  --score-mode neural_only
```

---

## 3. Overall 完整消融结果

| Setting | MRR | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Coverage | Avg Candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.352678 | 0.491130 | 0.376180 | 0.554622 | 0.396800 | 0.605976 | 0.409829 | 0.760971 | 250.458450 |
| Retrieval Only | 0.324618 | 0.448179 | 0.342564 | 0.521942 | 0.366394 | 0.590103 | 0.383624 | 0.760971 | 250.458450 |
| Neural Only | 0.091000 | 0.114846 | 0.077763 | 0.187675 | 0.101045 | 0.288515 | 0.126240 | 0.760971 | 250.458450 |
| w/o Long Prior | 0.355081 | 0.491130 | 0.377816 | 0.554622 | 0.398384 | 0.606909 | 0.411704 | 0.760971 | 250.458450 |
| w/o Topology | 0.348432 | 0.478058 | 0.368452 | 0.556489 | 0.394215 | 0.593838 | 0.403694 | 0.760971 | 250.458450 |
| w/o Semantic | 0.350222 | 0.477124 | 0.370226 | 0.544351 | 0.392191 | 0.591036 | 0.404057 | 0.760971 | 250.458450 |
| w/o Spatial | 0.359922 | 0.485528 | 0.379242 | 0.558357 | 0.402774 | 0.612512 | 0.416399 | 0.760971 | 250.458450 |

### 相对 Full 的变化

- **Retrieval Only vs Full**
  - MRR: `-0.028060`
  - HR@5: `-0.042951`
  - NDCG@10: `-0.030406`

- **Neural Only vs Full**
  - MRR: `-0.261678`
  - HR@5: `-0.376284`
  - NDCG@10: `-0.295755`

- **w/o Long Prior vs Full**
  - MRR: `+0.002403`
  - HR@5: `+0.000000`
  - NDCG@10: `+0.001584`

- **w/o Topology vs Full**
  - MRR: `-0.004246`
  - HR@5: `-0.013072`
  - NDCG@10: `-0.002585`

- **w/o Semantic vs Full**
  - MRR: `-0.002456`
  - HR@5: `-0.014006`
  - NDCG@10: `-0.004609`

- **w/o Spatial vs Full**
  - MRR: `+0.007244`
  - HR@5: `-0.005602`
  - NDCG@10: `+0.005974`

可以看到：

1. **Retrieval Only 与 Full 的差距不算大**，说明当前 NYC CoMa 设置下，检索先验本身已经提供了较强排序能力。
2. **Neural Only 明显低于 Retrieval Only**，说明若完全移除检索先验加成，仅靠神经匹配分数进行候选重排，当前模型还不能单独支撑最终效果。
3. **四个表征模块级消融的波动都明显小于 Retrieval/Neural 两类打分消融**，说明当前系统性能首先受“检索先验 + 打分融合”主导，其次才是单一模态编码器是否存在。
4. 在这轮 one-epoch NYC CoMa 实验中，`w/o Spatial` 与 `w/o Long Prior` 没有下降，甚至略高于 Full；这更像是**短周期训练波动**或当前融合冗余较大，而不能直接解释为这两个模块“无用”。

---

## 4. Warm / Cold 结果

### Warm targets

| Setting | warm_MRR | warm_HR@5 | warm_NDCG@5 | warm_HR@10 | warm_NDCG@10 | warm_HR@20 | warm_NDCG@20 | warm_Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.360763 | 0.502388 | 0.384803 | 0.567335 | 0.405896 | 0.619866 | 0.419224 | 0.778415 |
| Retrieval Only | 0.332059 | 0.458453 | 0.350417 | 0.533906 | 0.374793 | 0.603629 | 0.392418 | 0.778415 |
| Neural Only | 0.093086 | 0.117479 | 0.079546 | 0.191977 | 0.103362 | 0.295129 | 0.129134 | 0.778415 |
| w/o Long Prior | 0.363221 | 0.502388 | 0.386477 | 0.567335 | 0.407516 | 0.620821 | 0.421142 | 0.778415 |
| w/o Topology | 0.356419 | 0.489016 | 0.376898 | 0.569245 | 0.403252 | 0.607450 | 0.412947 | 0.778415 |
| w/o Semantic | 0.358250 | 0.488061 | 0.378713 | 0.556829 | 0.401181 | 0.604585 | 0.413320 | 0.778415 |
| w/o Spatial | 0.368173 | 0.496657 | 0.387935 | 0.571156 | 0.412007 | 0.626552 | 0.425944 | 0.778415 |

### Cold targets

三种设置下 cold 指标都为 0：

- `cold_candidate_target_coverage = 0.000000`
- `cold_MRR = 0.000000`
- `cold_HR@5/10/20 = 0.000000`

这说明在当前 `closed_world` 候选协议下，cold target 根本不在候选集合内，因此无论是 Retrieval Only、Neural Only 还是 Full reranker，都无法命中这部分样本。

换句话说，这组消融主要反映的是 **warm 子集上的排序贡献拆分**。

---

## 5. 结论

### 5.1 神经 reranker 的真实贡献

从 **Full vs Retrieval Only** 看：

- Full 的 `NDCG@10 = 0.396800`
- Retrieval Only 的 `NDCG@10 = 0.366394`
- 绝对提升为 **+0.030406**

说明神经 reranker **确实带来稳定增益**，但它的作用更像是在一个已经较强的候选先验之上做进一步精排，而不是从零建立排序能力。

### 5.2 检索先验的真实贡献

从 **Retrieval Only vs Neural Only** 看：

- Retrieval Only 的 `NDCG@10 = 0.366394`
- Neural Only 的 `NDCG@10 = 0.101045`
- 差值达到 **+0.265349**

说明在当前 NYC CoMa 设定中，**统计检索先验是主导性贡献来源**。候选检索分数不仅负责“把 target 找进候选集”，还已经编码了很强的排序偏好；相比之下，纯神经分数单独使用时表现明显不足。

### 5.3 四个表征模块的贡献观察

从本轮新增四个 `w/o` 消融看：

- **w/o Topology**：`NDCG@10 = 0.394215`，相对 Full 下降 `0.002585`
- **w/o Semantic**：`NDCG@10 = 0.392191`，相对 Full 下降 `0.004609`
- **w/o Long Prior**：`NDCG@10 = 0.398384`，相对 Full 上升 `0.001584`
- **w/o Spatial**：`NDCG@10 = 0.402774`，相对 Full 上升 `0.005974`

这说明：

1. **Topology 与 Semantic 在当前设置下有轻微正贡献**，移除后指标小幅下降。
2. **Long Prior 与 Spatial 在当前 one-epoch 结果中没有显示出稳定增益**；但由于提升/下降幅度都很小，这更可能说明这些模块的收益尚未在当前训练轮数下充分显现，或者与其他先验存在冗余，而不是可以直接判定为负贡献。
3. 相比之下，`Retrieval Only` / `Neural Only` 的差异远大于这四个模块级消融，说明**系统主效应仍来自候选检索先验与最终融合打分机制**。

### 5.4 对当前系统的整体判断

综合来看，当前 NYC CoMa 版本更接近一种：

> **强候选检索 + 神经精排修正**

而不是：

> **弱候选检索 + 强神经排序主导**

因此，如果后续想进一步提升结果，优先级可能是：

1. 继续优化检索与候选覆盖；
2. 再增强神经排序分数与检索先验之间的融合方式；
3. 若希望证明模型本身的“独立排序能力”，则需要继续提升 `model_scores` 的可分性，而不是只依赖 retrieval bias。

---

## 6. 本次训练与评估补充

新增四个 NYC CoMa 消融配置：

- `configs/nyc_v8_coma_wo_long_prior.yaml`
- `configs/nyc_v8_coma_wo_topology.yaml`
- `configs/nyc_v8_coma_wo_semantic.yaml`
- `configs/nyc_v8_coma_wo_spatial.yaml`

对应训练输出目录：

- `runs/prior_conditioned/NYC/v8_coma_wo_long_prior/`
- `runs/prior_conditioned/NYC/v8_coma_wo_topology/`
- `runs/prior_conditioned/NYC/v8_coma_wo_semantic/`
- `runs/prior_conditioned/NYC/v8_coma_wo_spatial/`

四个训练都成功完成，并产出 `best.pt`。

---

## 7. 本次代码改动

为支持该消融，我对评估流程做了最小改动：

- `poi_rec/training/train.py`
  - `evaluate_model(...)` 新增 `score_mode` 参数
  - 支持 `full / retrieval_only / neural_only`

- `poi_rec/training/evaluate.py`
  - `evaluate_checkpoint(...)` 透传 `score_mode`

- `scripts/evaluate.py`
  - 新增 CLI 参数 `--score-mode`

这样后续可以直接复用同一套评估入口，对其他城市或 checkpoint 继续做同类消融。