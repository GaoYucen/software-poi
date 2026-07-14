# 代价感知的自适应级联多索引检索

## 1. 技术名称

**中文名称：** 代价感知的自适应级联多索引检索  
**英文名称：** Cost-Aware Adaptive Cascaded Multi-Index Retrieval  
**英文缩写：** CACR

该技术用于 `nofix` 框架中的下一 POI 候选池构建阶段。其目标是在尽量维持候选覆盖率和最终推荐性能的前提下，减少不必要的索引访问、候选合并与精排开销。

---

## 2. 研究动机

当前候选生成器针对每个推荐查询，会从多种候选来源中分别检索候选，包括：

- 历史访问候选；
- 当前 POI 的转移候选；
- 更早历史 POI 的转移候选；
- 用户偏好候选；
- 时间候选；
- 类别候选；
- 空间近邻候选；
- 流行度候选；
- 可选的轨迹邻域候选。

现有实现对不同查询采用基本一致的索引访问计划，即无论当前查询是否已经通过低代价索引获得足够高质量的候选，仍继续访问空间、历史扩展或轨迹邻域等候选来源。

这种固定执行方式存在以下问题：

1. **索引访问冗余。** 对于转移规律明显或用户历史充分的查询，低代价索引已经可以产生较可靠的候选，不必继续访问全部来源。
2. **查询代价不稳定。** 空间近邻检索、历史扩展检索及轨迹邻域检索可能产生更高的计算和候选合并开销。
3. **候选数量增加不等于覆盖率显著提高。** 简单扩大候选池可能带来更大的精排开销，但候选覆盖率提升有限。
4. **缺乏数据库式查询计划。** 当前候选生成更接近固定流水线，没有根据中间结果动态决定后续索引访问顺序和是否提前终止。

因此，CACR 将候选检索建模为一个**自适应多索引查询处理问题**：先执行低代价索引查询，再根据中间候选结果判断是否需要访问高代价索引。

---

## 3. 核心思想

CACR 将多源候选召回划分为两个级联阶段：

### 3.1 第一阶段：低代价索引检索

优先访问查询开销较低、能够通过字典、邻接表或倒排索引直接获取结果的候选来源：

\[
\mathcal{M}_{\mathrm{cheap}}=
\{
\text{history},
\text{transition},
\text{user},
\text{temporal},
\text{category},
\text{popularity}
\}.
\]

对于查询 \(q\)，第一阶段候选集合为：

\[
\mathcal{C}^{(1)}_q=
\bigcup_{m\in\mathcal{M}_{\mathrm{cheap}}}
R_m(q),
\]

其中，\(R_m(q)\) 表示来源 \(m\) 针对查询 \(q\) 返回的 Top-N 候选。

第一阶段仍然沿用现有候选生成器中的处理逻辑：

1. 来源内分数归一化；
2. 来源权重加权；
3. 以 POI ID 为主键去重；
4. 对重复候选累加检索分数；
5. 合并来源位掩码；
6. 按融合分数排序。

### 3.2 第二阶段：按需访问高代价索引

将以下来源视为高代价或扩展性来源：

\[
\mathcal{M}_{\mathrm{expensive}}=
\{
\text{spatial},
\text{history-transition},
\text{trajectory-neighbor}
\}.
\]

只有当第一阶段候选结果不足或不确定性较高时，才执行第二阶段检索。

第二阶段候选集合为：

\[
\mathcal{C}^{(2)}_q=
\bigcup_{m\in\mathcal{M}_{\mathrm{expensive}}}
R_m(q).
\]

最终候选集合为：

\[
\mathcal{C}_q=
\begin{cases}
\operatorname{TopK}(\mathcal{C}^{(1)}_q),
& \operatorname{Expand}(q)=0,\\[4pt]
\operatorname{TopK}(\mathcal{C}^{(1)}_q\cup\mathcal{C}^{(2)}_q),
& \operatorname{Expand}(q)=1.
\end{cases}
\]

---

## 4. 自适应扩展判定

### 4.1 最简可验证版本

为了快速实现和验证，第一版只依据第一阶段候选数量决定是否继续检索：

\[
\operatorname{Expand}(q)=
\mathbb{I}
\left[
|\mathcal{C}^{(1)}_q|<K_{\min}
\right].
\]

其中：

- \(|\mathcal{C}^{(1)}_q|\) 为去重后的第一阶段候选数量；
- \(K_{\min}\) 为候选充分性阈值；
- \(\mathbb{I}[\cdot]\) 为指示函数。

例如，当最终候选池规模为 128 时，可设置：

```yaml
minimum_candidates: 128
```

该版本实现最简单，能够直接验证“候选充分时提前终止”是否可以降低在线检索开销。

### 4.2 推荐的完整版本

候选数量只能判断“是否足够多”，不能判断“是否足够可靠”。因此，可以进一步利用第一阶段融合分数的归一化熵衡量候选不确定性。

首先对第一阶段候选分数做 softmax 归一化：

\[
\bar{s}_p=
\frac{\exp(S(p\mid q)/T)}
{\sum_{p'\in\mathcal{C}^{(1)}_q}
\exp(S(p'\mid q)/T)},
\]

其中：

- \(S(p\mid q)\) 为现有多源融合检索分数；
- \(T\) 为温度参数，默认可设为 1。

归一化熵定义为：

\[
H(q)=
-\frac{
\sum_{p\in\mathcal{C}^{(1)}_q}
\bar{s}_p\log(\bar{s}_p+\epsilon)
}
{\log(|\mathcal{C}^{(1)}_q|+\epsilon)}.
\]

其中 \(H(q)\in[0,1]\)：

- 熵较低：少数候选分数明显更高，结果较集中；
- 熵较高：大量候选分数相近，第一阶段结果不确定。

完整扩展条件为：

\[
\operatorname{Expand}(q)=
\mathbb{I}
\left[
|\mathcal{C}^{(1)}_q|<K_{\min}
\;\lor\;
H(q)>\tau
\right],
\]

其中 \(\tau\) 为熵阈值。

推荐的初始搜索范围为：

\[
\tau\in\{0.5,0.6,0.7,0.8,0.9\}.
\]

---

## 5. 数据库技术特色

CACR 的主要贡献不在于增加新的候选来源，而在于优化多索引候选检索的执行方式。它可对应到以下数据库技术概念。

| 数据库技术概念 | CACR 中的对应设计 |
|---|---|
| 多索引查询 | 同时利用转移、用户、时间、类别、空间等索引 |
| 查询执行计划 | 先访问低代价索引，再决定是否访问高代价索引 |
| 自适应查询处理 | 根据中间候选集合动态改变后续执行路径 |
| 算子代价 | 统计不同候选来源的查询时间、返回规模和合并代价 |
| 提前终止 | 候选充分且置信度较高时跳过第二阶段 |
| Top-K 查询 | 最终只保留固定规模的候选集合 |
| 效果约束下的查询优化 | 在候选覆盖率基本不下降的条件下降低在线延迟 |
| 中间结果统计 | 使用候选数量、分数熵、来源数量评估查询状态 |

因此，该技术可表述为：

> 将下一 POI 候选生成建模为多索引 Top-K 查询，并根据中间候选结果动态调整索引访问计划，在候选覆盖率约束下减少冗余索引访问。

---

## 6. 与现有 nofix 框架的衔接

CACR 只修改候选生成阶段，不需要改变：

- 多模态表示学习模块；
- 序列骨干网络；
- Candidate Memory；
- 最终精排模块；
- 原有损失函数；
- 原有训练和评估协议。

因此，它具有以下优点：

1. 改动范围小；
2. 不会破坏现有模型主体；
3. 可以先脱离模型训练独立验证；
4. 可以复用现有 checkpoint 进行快速推理测试；
5. 容易设计清晰的效率—效果对比实验。

---

## 7. 代码改造方案

### 7.1 原有候选生成逻辑

当前逻辑可以概括为：

```python
add("history", ...)
add("user", ...)
add("transition", ...)
add("spatial", ...)
add("category", ...)
add("history_transition", ...)
add("temporal", ...)
add("trajectory_neighbor", ...)
add("popularity", ...)
```

### 7.2 修改后的级联逻辑

```python
# Stage 1: cheap indexes
add("history", ...)
add("user", ...)
add("transition", ...)
add("category", ...)
add("temporal", ...)
add("popularity", ...)

stage1_count = len(scores)
stage1_entropy = normalized_entropy(scores)

need_expand = (
    stage1_count < minimum_candidates
    or stage1_entropy > entropy_threshold
)

# Stage 2: expensive or expansive indexes
if need_expand:
    add("spatial", ...)
    add("history_transition", ...)

    if trajectory_enabled:
        add("trajectory_neighbor", ...)

# Final fusion, deduplication and Top-K truncation
final_candidates = topk(scores, num_candidates)
```

### 7.3 建议增加的统计信息

```python
retrieval_stats = {
    "expanded": need_expand,
    "num_stage1_candidates": stage1_count,
    "num_final_candidates": len(final_candidates),
    "stage1_entropy": stage1_entropy,
    "spatial_invoked": spatial_invoked,
    "history_transition_invoked": history_transition_invoked,
    "trajectory_neighbor_invoked": trajectory_neighbor_invoked,
    "stage1_time_ms": stage1_time_ms,
    "stage2_time_ms": stage2_time_ms,
    "merge_time_ms": merge_time_ms,
    "total_retrieval_time_ms": total_time_ms,
}
```

### 7.4 建议增加的配置项

```yaml
adaptive_cascade:
  enabled: true

  # 第一阶段候选数量下限
  minimum_candidates: 128

  # 是否启用熵判定
  use_entropy: true

  # 第一阶段分数熵阈值
  entropy_threshold: 0.75

  cheap_sources:
    - history
    - user
    - transition
    - category
    - temporal
    - popularity

  expensive_sources:
    - spatial
    - history_transition
    - trajectory_neighbor
```

---

## 8. 实验验证目标

实验需要回答以下三个核心问题：

### RQ1：CACR 是否能够减少候选检索开销？

重点比较：

- 平均候选检索延迟；
- P95 候选检索延迟；
- 高代价索引调用率；
- 平均访问候选来源数；
- 每秒处理查询数。

### RQ2：CACR 是否能够维持候选覆盖率？

重点比较：

- Candidate Recall@K；
- warm POI Candidate Recall@K；
- cold/unseen POI Candidate Recall@K；
- 不同用户历史长度下的候选覆盖率。

### RQ3：CACR 是否会影响最终推荐性能？

重点比较：

- HR@10；
- NDCG@10；
- 验证集总推理时间；
- 单位时间完成的查询数量。

---

## 9. 实验一：候选检索离线评估

### 9.1 实验目的

不重新训练模型，仅运行候选生成器，快速评估不同候选检索策略的覆盖率和效率。

### 9.2 对比方法

至少比较以下三种方法：

1. **Full-Retrieval**：当前固定访问全部候选来源的方法；
2. **Cheap-Only**：只执行第一阶段低代价索引；
3. **CACR**：根据候选数量和分数熵决定是否执行第二阶段。

可增加一个随机对照：

4. **Random-Skip**：按照与 CACR 相同的第二阶段调用比例随机跳过高代价索引。

`Random-Skip` 用于证明效率提升不是简单减少索引调用带来的，而是自适应判定优于随机跳过。

### 9.3 实验数据

使用验证集或测试集中的所有查询样本。对每个查询记录：

- 真实下一 POI；
- 第一阶段候选数量；
- 第一阶段分数熵；
- 是否执行第二阶段；
- 最终候选集合；
- 各阶段检索耗时；
- 候选来源调用情况。

### 9.4 核心指标

#### 候选覆盖率

\[
\mathrm{CandidateRecall@K}=
\frac{1}{|\mathcal{Q}|}
\sum_{q\in\mathcal{Q}}
\mathbb{I}[y_q\in\mathcal{C}_q],
\]

其中：

- \(\mathcal{Q}\) 为查询集合；
- \(y_q\) 为查询 \(q\) 的真实下一 POI；
- \(\mathcal{C}_q\) 为最终候选池。

#### 平均检索延迟

\[
\mathrm{AvgLatency}=
\frac{1}{|\mathcal{Q}|}
\sum_{q\in\mathcal{Q}}T(q).
\]

#### P95 检索延迟

将所有查询延迟排序，报告第 95 百分位数。

#### 高代价索引调用率

\[
\mathrm{ExpensiveCallRate}=
\frac{1}{|\mathcal{Q}|}
\sum_{q\in\mathcal{Q}}
\operatorname{Expand}(q).
\]

#### 平均访问来源数

\[
\mathrm{AvgSources}=
\frac{1}{|\mathcal{Q}|}
\sum_{q\in\mathcal{Q}}
N_{\mathrm{source}}(q).
\]

#### 吞吐量

\[
\mathrm{QPS}=
\frac{|\mathcal{Q}|}
{\mathrm{TotalRetrievalTime}}.
\]

### 9.5 推荐结果表

| 方法 | Candidate Recall@128 | Avg. Latency / ms | P95 Latency / ms | Expensive Call Rate | Avg. Sources | QPS |
|---|---:|---:|---:|---:|---:|---:|
| Full-Retrieval |  |  |  | 100% |  |  |
| Cheap-Only |  |  |  | 0% |  |  |
| Random-Skip |  |  |  |  |  |  |
| CACR |  |  |  |  |  |  |

### 9.6 期望结论

理想结果为：

- CACR 的 Candidate Recall@128 与 Full-Retrieval 基本持平；
- CACR 的平均延迟和 P95 延迟明显低于 Full-Retrieval；
- CACR 的高代价索引调用率显著低于 100%；
- CACR 的覆盖率明显高于具有相同调用率的 Random-Skip；
- Cheap-Only 延迟最低，但候选覆盖率下降明显。

---

## 10. 实验二：阈值敏感性分析

### 10.1 候选数量阈值

测试：

\[
K_{\min}\in\{64,96,128,160\}.
\]

观察：

- Candidate Recall@128；
- 平均延迟；
- 第二阶段调用率。

### 10.2 熵阈值

测试：

\[
\tau\in\{0.5,0.6,0.7,0.8,0.9\}.
\]

熵阈值越低，越容易触发第二阶段，覆盖率通常更高，但延迟也更高。

### 10.3 Pareto 曲线

建议绘制效果—效率 Pareto 曲线：

- 横轴：平均候选检索延迟；
- 纵轴：Candidate Recall@128；
- 每个点：一个不同的 \(\tau\)；
- 另标出 Full-Retrieval 和 Cheap-Only。

该图能够直观展示 CACR 在覆盖率和延迟之间的折中关系。

### 10.4 推荐选择原则

可以使用以下经验约束选择默认阈值：

\[
\Delta\mathrm{CandidateRecall@128}\geq-0.2\%,
\]

同时尽可能满足：

\[
\Delta\mathrm{AvgLatency}\leq-10\%.
\]

如果延迟下降可以达到 20% 以上，而候选覆盖率下降小于 0.2 个百分点，则实验结果较有说服力。

---

## 11. 实验三：模块消融实验

为了验证各判定信号的作用，比较：

1. **Full-Retrieval**：全部来源；
2. **Count-Only**：只使用候选数量条件；
3. **Entropy-Only**：只使用熵条件；
4. **Count+Entropy**：使用完整 CACR；
5. **Random-Skip**：随机决定是否执行第二阶段。

推荐表格：

| 方法 | Candidate Recall@128 | Avg. Latency | P95 Latency | Expensive Call Rate |
|---|---:|---:|---:|---:|
| Full-Retrieval |  |  |  | 100% |
| Count-Only |  |  |  |  |
| Entropy-Only |  |  |  |  |
| Random-Skip |  |  |  |  |
| Count+Entropy |  |  |  |  |

期望结果：

- Count-Only 能够过滤候选数量明显充足的简单查询；
- Entropy-Only 能够识别候选较多但分数分布不确定的困难查询；
- Count+Entropy 在覆盖率和效率之间取得最佳折中；
- Random-Skip 在相同调用率下覆盖率更低。

---

## 12. 实验四：最终推荐性能验证

### 12.1 快速验证

不重新训练模型，直接加载现有 `nofix` checkpoint，仅替换候选生成器，执行验证集或测试集推理。

比较：

- Full-Retrieval；
- CACR。

报告：

| 方法 | Candidate Recall@128 | HR@10 | NDCG@10 | Retrieval Time | Total Inference Time |
|---|---:|---:|---:|---:|---:|
| Full-Retrieval |  |  |  |  |  |
| CACR |  |  |  |  |  |

如果 HR@10 与 NDCG@10 基本不变，而检索时间或总推理时间下降，则能够证明 CACR 可以无缝替换现有候选生成方式。

### 12.2 正式验证

在确定默认阈值后，使用 CACR 重新训练一次完整模型，并与原始 nofix 进行最终对比。

正式实验应报告：

- Candidate Recall@128；
- HR@5、HR@10；
- NDCG@5、NDCG@10；
- 平均候选检索时间；
- P95 候选检索时间；
- 总训练时间；
- 总推理时间。

---

## 13. 分组分析

为了进一步解释 CACR 在什么场景下有效，可以按照查询难度进行分组。

### 13.1 按用户历史长度分组

- 短序列用户；
- 中等序列用户；
- 长序列用户。

预期：

- 长序列用户通常能够通过历史和转移索引获得较充分候选，第二阶段调用率较低；
- 短序列用户更依赖空间、类别和流行度等扩展来源。

### 13.2 按转移分布熵分组

- 低熵查询：下一 POI 转移较集中；
- 高熵查询：下一 POI 转移较分散。

预期：

- 低熵查询更容易提前终止；
- 高熵查询更频繁触发第二阶段。

### 13.3 按 POI 热度分组

- 热门 POI；
- 中等热度 POI；
- 长尾 POI。

该实验可以验证 CACR 是否会因为提前终止而损害长尾 POI 的候选覆盖率。

推荐额外报告：

\[
\mathrm{LongTailRecall@K}.
\]

---

## 14. 效率测量注意事项

### 14.1 预热

在正式计时前执行若干批次预热，以减少首次调用、缓存初始化和内存分配的影响。

### 14.2 缓存设置

应分别报告：

- 冷缓存场景；
- 热缓存场景。

尤其对于空间候选缓存，冷缓存和热缓存的开销可能差异较大。

### 14.3 同步计时

如果候选生成涉及 GPU 运算，需要在计时前后进行设备同步，否则测得时间可能不准确。

### 14.4 重复实验

每个实验至少重复 3 次，报告平均值和标准差。

### 14.5 统一批大小

所有方法必须使用相同：

- batch size；
- 候选池最终大小；
- 硬件环境；
- 数据划分；
- checkpoint；
- 评估协议。

---

## 15. 统计显著性

对于最终 HR@10 和 NDCG@10，可使用：

- 配对 bootstrap；
- 配对 t 检验；
- Wilcoxon 符号秩检验。

对于查询延迟，建议报告：

- 平均值；
- 中位数；
- P95；
- 标准差。

查询延迟通常呈非正态分布，因此 P95 和中位数比仅报告平均值更有数据库系统意义。

---

## 16. 成功判定标准

CACR 可以被认为有效，建议至少满足以下条件：

1. Candidate Recall@128 相比 Full-Retrieval 下降不超过 0.2 个百分点；
2. HR@10 和 NDCG@10 基本不下降；
3. 平均候选检索延迟下降至少 10%；
4. P95 候选检索延迟有所下降；
5. 高代价索引调用率低于 70%；
6. 在与 CACR 相同调用率下，Random-Skip 的覆盖率低于 CACR。

如果能够达到：

- 延迟下降 20% 以上；
- 高代价索引调用率下降至 40%–60%；
- Candidate Recall@128 下降小于 0.2 个百分点；

则该技术点已经具备较完整的实验支撑。

---

## 17. 论文中的方法描述建议

### 17.1 技术问题

> 现有多模态候选生成方法通常为所有查询执行固定的索引访问计划，忽略了不同查询在用户历史完整度、转移确定性和候选充分性方面的差异，导致空间和轨迹等高代价索引被重复访问，增加了在线候选检索开销。

### 17.2 方法概述

> 为此，本文提出代价感知的自适应级联多索引检索方法。该方法首先访问转移、用户、时间、类别和流行度等低代价索引，并根据中间候选集合的规模及检索分数熵评估当前结果的充分性。当候选不足或分数分布不确定时，方法进一步访问空间、历史扩展及轨迹邻域索引；否则提前终止检索。通过动态调整索引访问计划，该方法在维持目标 POI 候选覆盖率的同时减少冗余索引访问。

### 17.3 创新点表述

> 提出一种面向下一 POI 推荐的代价感知自适应级联检索机制，将多模态候选生成建模为多索引 Top-K 查询，并利用中间结果统计动态决定高代价索引的访问与提前终止，从而实现候选覆盖率与在线检索效率之间的平衡。

---

## 18. 论文中的实验问题表述

可以在实验章节设置以下研究问题：

- **RQ1：** CACR 能否在维持候选覆盖率的同时降低多索引候选检索延迟？
- **RQ2：** 候选数量和分数熵是否能够有效识别需要扩展检索的困难查询？
- **RQ3：** CACR 是否能够在不损害 HR 和 NDCG 的情况下提高端到端推理效率？
- **RQ4：** 不同候选数量阈值和熵阈值对覆盖率—效率折中的影响如何？

---

## 19. 最小实现与实验路径

为了尽快完成验证，建议按照以下顺序推进。

### 第一步：实现 Count-Only

只加入：

```python
need_expand = len(stage1_candidates) < minimum_candidates
```

首先验证提前终止是否能降低检索时间。

### 第二步：加入统计日志

记录：

- 第一阶段候选数量；
- 是否扩展；
- 各阶段耗时；
- 最终候选是否包含真实目标。

### 第三步：离线扫描阈值

测试：

```text
minimum_candidates ∈ {64, 96, 128, 160}
```

不需要重新训练模型。

### 第四步：加入 Entropy

在 Count-Only 有效后，再加入分数熵条件，比较：

- Count-Only；
- Entropy-Only；
- Count+Entropy。

### 第五步：复用 checkpoint 验证 HR/NDCG

直接替换候选生成器进行推理，确认最终推荐指标基本不下降。

### 第六步：选定配置后重新训练

仅对最终选定的 CACR 配置进行一次完整训练，形成正式实验结果。

---

## 20. 建议的最终技术定位

CACR 不应被包装为一个新的推荐模型模块，而应定位为：

> **面向多模态下一 POI 推荐的自适应多索引候选查询优化方法。**

其主要价值是：

1. 把候选召回从固定预处理提升为可优化的数据库查询阶段；
2. 通过动态查询计划减少高代价索引访问；
3. 显式分析候选覆盖率、索引调用率和在线延迟之间的关系；
4. 在不改变 nofix 主模型的情况下完成快速实验验证；
5. 为《软件学报》多模态数据管理与融合专刊提供更明确的数据库技术主线。
