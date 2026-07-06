# Path-LLM 思路梳理与 Next POI Recommendation 迁移方案

> 本文用于本地阅读与后续实现。内容分为两部分：  
> 1. **Path-LLM 原论文到底做了什么**；  
> 2. **如何把它迁移到 Next POI Recommendation，并形成不只是“换任务”的新方法**。

---

## 1. Path-LLM 的研究问题

论文标题：

**Path-LLM: A Multi-Modal Path Representation Learning by Aligning and Fusing with Large Language Models**

论文要解决的不是单一的 ETA 预测，而是更上层的：

> **多模态路径表示学习（Multi-Modal Path Representation Learning）**

传统路径表示学习方法通常只使用路网拓扑，例如：

- 哪些道路相连；
- 道路之间的结构关系；
- 一条路径由哪些 road segments 顺序组成。

但实际道路还具有大量非拓扑信息，例如：

- road name；
- road type；
- road length；
- 道路的区域和功能属性。

因此，Path-LLM 的基本观点是：

> 一条路径既是一个**拓扑序列**，也是一个**语义序列**。  
> 高质量路径表示应该同时理解“它在路网上怎么走”以及“这些道路分别是什么”。

---

# 2. 一句话理解 Path-LLM

Path-LLM 可以概括为：

```text
Topology Modality
        +
Textual Modality
        ↓
Cross-Modal Alignment
        ↓
Adaptive Fusion
        ↓
GPT-2 Sequence Modeling
        ↓
General Path Representation
        ↓
Downstream Tasks
```

更直观地说：

> **把一条路径看成一句话，把每个 road segment 看成一个 domain token；每个 token 同时具有拓扑表示和文本表示。先对齐两个模态，再自适应融合，然后把融合后的连续向量直接送入 GPT-2。**

这也是理解整篇论文最重要的一句话。

---

# 3. Path-LLM 的输入是如何构造的

假设一条路径为：

\[
P = (e_1, e_2, \ldots, e_L)
\]

其中每个 \(e_i\) 是一个 road segment。

Path-LLM 为同一个 road segment 构造两种表示。

---

## 3.1 拓扑模态

对每个 road segment 使用：

```text
Node2Vec
   ↓
Feed-Forward Network
   ↓
Topological Embedding
```

得到：

\[
q_i^{topo}
\]

它主要表达：

- 该道路在路网中的结构位置；
- 与其他道路的连接关系；
- 所处局部路网结构；
- 在全局网络中的拓扑角色。

整条路径的拓扑序列表示为：

\[
Q_p =
[q_1^{topo}, q_2^{topo}, \ldots, q_L^{topo}]
\]

---

## 3.2 文本模态

论文为每个 road segment 构造文本描述，主要包括：

- road name；
- road type；
- road length。

然后使用文本 embedding 模型编码，得到：

\[
q_i^{text}
\]

整条路径的文本序列表示为：

\[
Q_t =
[q_1^{text}, q_2^{text}, \ldots, q_L^{text}]
\]

因此，同一个 road segment 同时对应：

\[
q_i^{topo}
\quad\text{和}\quad
q_i^{text}
\]

---

# 4. Path-LLM 的完整模型流程

完整流程可以写成：

```text
Road Segment Sequence
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
Road Network Topology           Road Attribute Text
        │                              │
    Node2Vec                    Text Embedding
        │                              │
       FFN                         Max Pooling
        │                              │
       Qp                             Qt
        │                              │
        └──────────────┬───────────────┘
                       ▼
                    TPalign
                       │
               Cross-Modal Alignment
                       │
                       ▼
                    TPfusion
                       │
                Adaptive Gating
                       │
                       ▼
                       Qf
                       │
                       ▼
             Partially Frozen GPT-2
                       │
                       ▼
              Path Representation
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
  Travel Time Estimation      Path Ranking
```

模型的四个核心模块为：

1. **Input Embeddings**
2. **TPalign**
3. **TPfusion**
4. **Partially Frozen GPT-2 + TOCL**

下面分别说明。

---

# 5. 核心模块一：TPalign

## 5.1 为什么不能直接融合

拓扑 embedding 和文本 embedding 来自完全不同的表示空间。

例如：

- 两条道路在路网上非常接近；
- 但它们的名称、道路类型和功能语义差异很大。

此时可能出现：

\[
\text{topology space 中很近}
\]

但：

\[
\text{text space 中很远}
\]

如果直接：

```text
Qp + Qt
```

或者：

```text
Concat(Qp, Qt)
```

模型需要同时承担两个任务：

1. 先消除模态间语义鸿沟；
2. 再进行真正的信息融合。

Path-LLM 的核心思想是：

> **先对齐，再融合。**

---

## 5.2 Instance-level Contrastive Learning

对同一个 road segment：

\[
(q_i^{topo}, q_i^{text})
\]

视为正样本。

不同 road segment：

\[
(q_i^{topo}, q_j^{text}),
\quad i \neq j
\]

视为负样本。

目标是：

```text
Matched Topology-Text Pair
        → Pull Closer

Unmatched Topology-Text Pair
        → Push Apart
```

因此，TPalign 首先解决：

> **同一个实体的不同模态是否可以在统一 embedding space 中对应起来。**

---

## 5.3 Feature-level Contrastive Learning

Path-LLM 不只在样本层面对齐，还进一步在 feature dimension 层面对齐。

可以简单理解为：

- instance-level：对齐“对象”；
- feature-level：对齐“表示空间的结构”。

最终：

\[
\mathcal{L}_{align}
=
\lambda \mathcal{L}_{ins}
+
(1-\lambda)\mathcal{L}_{fea}
\]

---

## 5.4 TPalign 的真正意义

TPalign 解决的不是“如何融合”，而是：

> **如何让 topology 和 text 先说同一种 embedding language。**

这是整个 Path-LLM 最值得迁移到其他多模态任务中的思想之一。

---

# 6. 核心模块二：TPfusion

即使完成了模态对齐，也不能假设所有 road segments 对两个模态的依赖程度都一样。

例如：

- 普通支路可能更依赖拓扑；
- 著名主干道可能更依赖道路名称和功能语义。

因此，Path-LLM 使用动态门控机制：

\[
G
=
\sigma(W_p Q_p + W_t Q_t + b)
\]

然后：

\[
Q_f
=
G \odot Q_p
+
(1-G) \odot Q_t
\]

其中：

- \(G\) 越大，越依赖 topology；
- \(G\) 越小，越依赖 text。

因此：

> **TPalign 解决“能不能融合”；  
> TPfusion 解决“每个位置应该融合多少”。**

---

# 7. 核心模块三：GPT-2

这是最容易误解的部分。

Path-LLM 不是：

```text
Prompt
  ↓
GPT-2
  ↓
生成文本答案
```

也不是让 GPT-2 直接输出 ETA。

它真正做的是：

```text
Multimodal Fused Embedding
        ↓
GPT-2 Transformer Layers
        ↓
Sequence Representation
```

---

## 7.1 绕过原始 Token Embedding

通常 GPT-2 输入为：

```text
Token ID
   ↓
Token Embedding
   ↓
Transformer
```

Path-LLM 改成：

```text
Road Multimodal Embedding
          ↓
   Skip Token Embedding
          ↓
      Transformer
```

即直接使用：

```python
inputs_embeds = Q_f
```

因此：

> **road segment 被重新定义成了 GPT-2 的 domain token。**

---

## 7.2 GPT-2 如何训练

论文采用部分冻结策略。

### 冻结

- Multi-Head Self-Attention；
- Feed-Forward Network。

### 训练

- Positional Embedding；
- LayerNorm。

目的为：

- 保留 GPT-2 预训练得到的通用序列建模能力；
- 降低训练成本；
- 避免小规模轨迹数据导致过拟合；
- 让位置编码适应道路序列。

因此，Path-LLM 的本质不是传统“调用大语言模型”，而是：

> **Domain Encoder + Cross-Modal Alignment + Adaptive Fusion + Frozen Pretrained Transformer**

---

# 8. 核心模块四：TOCL

TOCL 全称：

**Two-stage Overlapping Curriculum Learning**

它解决的问题是：

> 多模态样本的学习难度并不相同。

有些路径：

- topology 和 text 很一致；
- 很容易学习。

有些路径：

- 两个模态冲突明显；
- 下游预测也更困难。

因此，作者先根据任务误差给训练样本排序：

```text
Easy
 ↓
Medium
 ↓
Hard
```

---

## 8.1 Overlapping Curriculum

不是直接：

```text
Easy → Medium → Hard
```

而是使用有重叠的滑动窗口：

```text
Round 1: [easy................]
Round 2:       [................]
Round 3:             [..........hard]
```

相邻阶段部分样本重复。

作用是：

> 学习更难样本时，仍然不断复习前面已经学过的知识。

---

## 8.2 Two-stage Curriculum

第一阶段：

```text
Easy → Hard
```

第二阶段：

```text
再重新覆盖全部数据
```

目的为：

- 避免 catastrophic forgetting；
- 学会困难样本后重新巩固基础知识。

---

# 9. Path-LLM 的创新点总结

我认为可以分为三个层次。

---

## 9.1 第一层：最核心创新

### 把 road segment 重新定义为 LLM 的 domain token

```text
Road Segment
    ↓
Multimodal Embedding
    ↓
Domain Token
    ↓
Pretrained GPT-2
```

这是整个工作的基础。

---

## 9.2 第二层：模型级创新

### TPalign + TPfusion

即：

```text
Align First
    ↓
Fuse Later
```

而不是：

```text
Concat
  ↓
Transformer
```

这是最值得迁移到其他多模态任务的设计。

---

## 9.3 第三层：训练级创新

### TOCL

解决：

- 多模态样本难度不同；
- easy/hard sample 学习不均衡；
- catastrophic forgetting。

---

# 10. Path-LLM 的不足

如果准备基于它做新论文，需要清楚它的不足。

---

## 10.1 模态其实不算非常丰富

原论文主要是：

```text
Topology
+
Text
```

文本信息也主要来源于：

- road name；
- road type；
- road length。

严格来说，它更接近：

> 双模态路径表示学习。

---

## 10.2 GPT-2 的作用仍需进一步论证

Path-LLM 假设：

> GPT-2 学到的语言序列规律可以迁移到道路序列。

这是一个有趣假设，但理论上并不完全自然。

因此，迁移到新任务时不能只说：

> GPT-2 擅长序列建模，所以使用 GPT-2。

需要解释：

- 为什么预训练语言 Transformer 对目标序列有帮助；
- domain tokens 如何进入其表示空间；
- 为什么比普通 Transformer 更好。

---

## 10.3 TPalign 只针对 Topology-Text

如果扩展到三种甚至四种模态，就需要重新思考：

- pairwise alignment；
- anchor-based alignment；
- shared latent space；
- modality-specific preservation。

这恰好也是做 Next POI 时可以形成新创新的地方。

---

# 11. 将 Path-LLM 迁移到 Next POI Recommendation

Next POI Recommendation 的任务为：

给定用户历史访问序列：

\[
S_u^t
=
(p_1, p_2, \ldots, p_t)
\]

预测：

\[
p_{t+1}
\]

最自然的对应关系为：

| Path-LLM | Next POI |
|---|---|
| Road segment | POI |
| Path | Check-in sequence |
| Road topology | POI transition graph |
| Road text | POI semantics |
| Road sequence | User mobility sequence |
| Path representation | Mobility representation |
| Downstream task | Next POI prediction |

因此：

> **把 POI 看成新的 domain token。**

---

# 12. 不应该怎样做

最简单的迁移方式是：

```text
POI Transition Graph
        +
POI Category Text
        ↓
TPalign
        ↓
TPfusion
        ↓
GPT-2
        ↓
Next POI
```

这当然可以工作。

但问题是：

> 它太像“把 road segment 换成 POI”。

很容易被审稿人认为：

- method adaptation；
- direct application；
- 缺少任务特定创新。

因此，真正应该做的是：

> 在保留 Path-LLM 核心思想的基础上，针对 Next POI 的本质重新设计多模态结构。

---

# 13. Next POI 中真正需要的三类模态

我建议至少使用三种模态：

\[
\boxed{
Topology
+
Semantics
+
Temporal Dynamics
}
\]

---

## 13.1 模态一：POI Transition Topology

根据所有用户的连续访问记录：

```text
POI A → POI B
POI B → POI C
POI A → POI D
```

构建有向图：

\[
G=(V,E)
\]

其中：

- node：POI；
- edge：两个 POI 曾连续访问；
- edge weight：转移频率。

然后：

\[
z_i^{topo}
=
GraphEncoder(p_i)
\]

可选：

- Node2Vec；
- GraphSAGE；
- GAT；
- GCN。

为了与 Path-LLM 做最忠实的第一版复现，建议先使用：

```text
Node2Vec + FFN
```

之后再替换为图神经网络。

---

## 13.2 模态二：POI Semantic Text

对每个 POI 构造自然语言描述，例如：

```text
Name: Joe's Pizza
Category: Restaurant
Subcategory: Pizza Restaurant
Region: Manhattan
Typical Visit Time: Evening
```

然后：

\[
z_i^{sem}
=
TextEncoder(description_i)
\]

语义来源可以包括：

- POI category；
- subcategory；
- region；
- tag；
- user tips；
- 外部 POI metadata。

---

## 13.3 模态三：Temporal Dynamics

这是 Next POI 与路径表示学习最大的区别。

同一个用户在不同时间的下一步行为差异极大。

例如：

```text
08:00 → Office / Station
12:00 → Restaurant
18:00 → Restaurant / Station
23:00 → Home / Bar
```

因此，对第 \(i\) 次访问应构造：

\[
z_i^{time}
\]

它可以包括：

- hour-of-day；
- day-of-week；
- time interval；
- periodic pattern；
- stay duration；
- recent mobility rhythm。

---

# 14. 推荐的整体模型

我建议的整体结构为：

```text
Historical Check-in Sequence
[p1, p2, ..., pt]
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
 POI Transition Graph             POI Semantic Text
         │                              │
  Graph Encoder                   Text Encoder
         │                              │
      Q_topo                         Q_sem
         │                              │
         └────── Cross-Modal Alignment ┘
                      │
                      ▼
               Topology-Semantic
                    Alignment
                      │
                      ▼
               Adaptive Fusion
                      │
                      ▼
            Static POI Representation
                      │
                      ├──────────────┐
                      │              │
                      ▼              ▼
               Temporal Context   User Context
                      │              │
                      └──────┬───────┘
                             ▼
                  Dynamic POI Token
                             │
                             ▼
                      Frozen GPT-2
                             │
                             ▼
                Mobility Representation
                             │
                             ▼
                   Candidate Scoring
                             │
                             ▼
                       Top-K POIs
```

---

# 15. 推荐不要直接做三模态“平铺式融合”

最简单的做法是：

\[
Q
=
Q^{topo}
+
Q^{sem}
+
Q^{time}
\]

但我不推荐。

原因是三种模态的性质完全不同：

### Topology

是相对稳定的：

```text
Static Structural Knowledge
```

### Semantics

也是相对稳定的：

```text
Static Semantic Knowledge
```

### Time

是访问事件级动态变量：

```text
Dynamic Context
```

因此更合理的逻辑是：

## 第一阶段：构造静态 POI 表示

\[
z_i^{static}
=
Fuse(
z_i^{topo},
z_i^{sem}
)
\]

## 第二阶段：注入动态上下文

\[
z_{u,i}^{dynamic}
=
Contextualize(
z_i^{static},
z_i^{time},
z_u
)
\]

然后把：

\[
[z_{u,1}^{dynamic},\ldots,z_{u,t}^{dynamic}]
\]

送入 GPT-2。

这比简单三模态融合更符合 Next POI 的问题结构。

---

# 16. 对齐模块应该如何改

Path-LLM 做的是：

```text
Topology ↔ Text
```

你的版本可以首先保留：

```text
POI Transition Topology
          ↕
     POI Semantics
```

即：

\[
\mathcal{L}_{TS-align}
\]

正样本：

```text
同一个 POI 的 topology embedding
+
同一个 POI 的 semantic embedding
```

负样本：

```text
Topology of POI i
+
Semantics of POI j
```

---

## 16.1 更进一步：考虑 Transition-Semantic Discrepancy

真正值得做的创新可以是：

> 某些 POI 在转移图中的行为角色与其语义类别一致；  
> 某些 POI 则存在显著差异。

例如：

- 两个都是 Restaurant；
- 但一个位于商业区，是午餐节点；
- 一个位于住宅区，是夜间节点。

它们的：

```text
Semantic Role
```

类似，但：

```text
Mobility Transition Role
```

不同。

因此可以显式建模：

\[
D_i
=
Distance(
z_i^{topo},
z_i^{sem}
)
\]

并让 discrepancy 参与：

- fusion；
- curriculum；
- uncertainty estimation。

---

# 17. 推荐的融合模块

最基础版本仍然可以使用 Path-LLM 的 gating：

\[
G_i
=
\sigma(
W_t z_i^{topo}
+
W_s z_i^{sem}
+
b
)
\]

然后：

\[
z_i^{static}
=
G_i \odot z_i^{topo}
+
(1-G_i)\odot z_i^{sem}
\]

但是更推荐：

> gate 不只由两个模态决定，还应受到当前时间上下文影响。

即：

\[
G_{u,i}
=
\sigma(
W_t z_i^{topo}
+
W_s z_i^{sem}
+
W_c z_i^{time}
+
b
)
\]

这样：

> 同一个 POI 在不同时间条件下，对 topology 和 semantics 的依赖可以不同。

例如：

- 工作日早高峰，更依赖 transition pattern；
- 周末探索行为，更依赖 semantic preference。

这比 Path-LLM 的固定 POI-level fusion 更符合 Next POI。

---

# 18. GPT-2 在 Next POI 中应该怎样使用

最直接的方式仍然是：

```text
Dynamic POI Embeddings
        ↓
inputs_embeds
        ↓
GPT-2
```

即：

```python
outputs = gpt2(
    inputs_embeds=poi_tokens,
    attention_mask=attention_mask
)
```

不需要把 POI ID 变成自然语言 prompt。

---

## 18.1 每个 POI Token 包含什么

建议：

\[
x_i
=
z_i^{static}
+
z_i^{time}
+
z_u
\]

或者：

\[
x_i
=
Fusion(
z_i^{static},
z_i^{time},
z_u
)
\]

其中：

- \(z_i^{static}\)：POI topology + semantics；
- \(z_i^{time}\)：访问时间上下文；
- \(z_u\)：用户偏好表示。

最终：

```text
POI 1 Token
POI 2 Token
POI 3 Token
...
```

进入 GPT-2。

---

# 19. 输出层不要使用固定城市分类器

最简单的 Next POI 做法是：

\[
h_t
\rightarrow
MLP
\rightarrow
|\mathcal{P}| \text{-way classification}
\]

但这样存在两个问题：

1. 不适合跨城市；
2. 对 unseen POI 很弱。

更推荐：

\[
score(p \mid h_t)
=
h_t^\top z_p
\]

即：

```text
Sequence Representation
        ×
Candidate POI Representation
        ↓
Matching Score
```

这样：

- POI 本身有独立 representation；
- 支持 candidate ranking；
- 更适合 cross-city；
- 更适合 cold-start；
- 更符合 representation learning 思路。

---

# 20. 推荐的训练目标

建议至少包含三部分。

---

## 20.1 Next POI Prediction Loss

\[
\mathcal{L}_{next}
\]

可使用：

- Cross Entropy；
- Sampled Softmax；
- BPR Loss；
- InfoNCE Ranking Loss。

---

## 20.2 Cross-Modal Alignment Loss

\[
\mathcal{L}_{align}
\]

即：

```text
Topology ↔ Semantics
```

---

## 20.3 Sequence Modeling Loss

可以加入：

```text
Masked POI Modeling
```

或者：

```text
Next-Token POI Prediction
```

因此：

\[
\mathcal{L}
=
\mathcal{L}_{next}
+
\alpha \mathcal{L}_{align}
+
\beta \mathcal{L}_{seq}
\]

---

# 21. TOCL 应该怎样迁移

Path-LLM 使用预测误差衡量样本难度。

对于 Next POI，不应该直接照搬 MSE。

可以定义：

\[
Difficulty_i
=
-\log
P(p_{t+1}\mid p_{1:t})
\]

也可以进一步加入：

\[
Difficulty_i
=
\alpha
\cdot
PredictionDifficulty_i
+
\beta
\cdot
CrossModalDiscrepancy_i
\]

其中：

\[
CrossModalDiscrepancy_i
=
D(
z_i^{topo},
z_i^{sem}
)
\]

这样形成：

> **Cross-Modal Discrepancy-Aware Curriculum Learning**

训练顺序：

```text
High-consistency / Easy Transition
                ↓
Medium Difficulty
                ↓
Rare Transition
                ↓
Long-distance Jump
                ↓
Cold-start / Sparse POI
```

这比直接复用 Path-LLM 的 TOCL 更有任务针对性。

---

# 22. 数据集选择

## 首选：Foursquare NYC & Tokyo

这是最适合当前思路的数据集。

原因不是单纯“常用”，而是它可以同时支持：

### Topology

由 check-in sequence 构造：

```text
POI A → POI B
```

### Semantics

使用：

- POI category；
- POI subcategory。

### Time

使用：

- timestamp；
- hour；
- weekday；
- interval。

### Space

使用：

- latitude；
- longitude；
- spatial distance。

因此，它与你的模型结构非常匹配：

\[
\boxed{
Transition
+
Semantics
+
Temporal Context
+
Spatial Context
}
\]

---

## 不建议首先使用 Gowalla

Gowalla 的优点：

- 数据规模大；
- 适合轨迹和地理建模。

但对当前方法而言，最大的不足是：

> POI semantic information 较弱。

如果要构造丰富文本语义，往往还需要额外做：

```text
Location ID
    ↓
Reverse Geocoding
    ↓
OSM Matching
    ↓
POI Metadata
```

数据工程成本较高。

因此：

> **第一版建议使用 Foursquare NYC/Tokyo。**

---

# 23. 推荐的实验结构

## 23.1 Main Experiment

```text
Foursquare NYC
Foursquare Tokyo
```

任务：

```text
Next POI Recommendation
```

指标：

- Recall@K；
- NDCG@K；
- MRR；
- HitRate@K。

---

## 23.2 Ablation Study

至少包括：

```text
w/o Topology
w/o Semantics
w/o Time
w/o Alignment
w/o Dynamic Fusion
w/o GPT-2
w/o Curriculum
```

---

## 23.3 Few-shot Experiment

使用：

```text
5%
10%
30%
```

训练数据。

验证：

> 预训练 Transformer + 多模态知识是否在稀疏数据下更有效。

---

## 23.4 Cross-city Experiment

```text
NYC → Tokyo
Tokyo → NYC
```

但必须注意：

> 不建议使用固定 POI ID 分类头。

应该使用：

```text
Query Representation
        ×
Candidate POI Representation
```

进行匹配。

---

## 23.5 Cold-start Experiment

按 POI 历史频次划分：

```text
Head POIs
Tail POIs
Cold POIs
```

验证：

> semantic information 是否可以弥补 topology sparsity。

这会非常适合突出多模态融合的价值。

---

# 24. 推荐的研究问题

不建议写：

> 使用 GPT-2 解决 Next POI Recommendation。

也不建议写：

> 将 Path-LLM 扩展到 POI 推荐。

更好的核心问题是：

> **如何把 POI 的全局转移结构、语义属性和动态时序上下文对齐到统一的预训练序列模型表示空间，从而提升下一兴趣点推荐的准确性、稀疏场景鲁棒性和跨区域泛化能力？**

---

# 25. 推荐的三个创新点

## 创新点一：Transition-Semantic POI Representation Alignment

提出：

```text
POI Transition Topology
          ↕
     POI Semantics
```

的跨模态对齐机制。

目标：

- 消除 topology-semantic gap；
- 学习统一 POI representation；
- 提高稀疏 POI 表示质量。

---

## 创新点二：Context-Aware Dynamic Multimodal Fusion

不是固定地融合 topology 和 semantics。

而是：

\[
Gate
=
f(
Topology,
Semantics,
Time,
User Context
)
\]

使同一个 POI 在不同用户、不同时间下具有不同的模态权重。

这是相对于 Path-LLM 非常自然的任务特定扩展。

---

## 创新点三：Discrepancy-Aware Curriculum Learning

同时根据：

- prediction difficulty；
- topology-semantic discrepancy；

定义训练难度。

实现：

```text
Easy Consistent Samples
        ↓
Hard Conflicting Samples
```

逐步训练。

---

# 26. 推荐的论文主线

整篇论文最好围绕一个明确逻辑展开：

```text
Problem 1:
POI transition structure and POI semantics live in different spaces.
        ↓
Solution:
Cross-modal alignment.

Problem 2:
The importance of topology and semantics varies with temporal context.
        ↓
Solution:
Context-aware dynamic fusion.

Problem 3:
Cross-modal discrepancy creates heterogeneous sample difficulty.
        ↓
Solution:
Discrepancy-aware curriculum learning.
```

这样三个创新点之间是有因果关系的，而不是简单堆模块。

---

# 27. 推荐的最小可运行版本

第一版不要一次做得过于复杂。

建议顺序：

---

## Stage 1：复现 Path-LLM 式双模态结构

```text
POI Transition Graph
        +
POI Category Text
        ↓
Alignment
        ↓
Gated Fusion
        ↓
GPT-2
        ↓
Next POI
```

目标：

> 先确认 Path-LLM 思路迁移到 POI 任务是否有效。

---

## Stage 2：加入 Temporal Context

```text
Static POI Representation
        +
Time Embedding
        ↓
Dynamic POI Token
```

目标：

> 超过直接迁移版本。

---

## Stage 3：改造 Dynamic Fusion

让 gate 感知：

- time；
- user；
- recent context。

---

## Stage 4：加入 Curriculum

最后再实现：

```text
Discrepancy-Aware Curriculum
```

这样开发风险最低。

---

# 28. 推荐的代码模块划分

```text
project/
│
├── data/
│   ├── preprocess.py
│   ├── build_transition_graph.py
│   ├── build_poi_text.py
│   └── sequence_dataset.py
│
├── models/
│   ├── topology_encoder.py
│   ├── semantic_encoder.py
│   ├── temporal_encoder.py
│   ├── alignment.py
│   ├── fusion.py
│   ├── gpt_backbone.py
│   └── poi_model.py
│
├── losses/
│   ├── alignment_loss.py
│   ├── recommendation_loss.py
│   └── curriculum.py
│
├── trainers/
│   ├── pretrain_alignment.py
│   └── train_next_poi.py
│
└── experiments/
    ├── main.py
    ├── ablation.py
    ├── fewshot.py
    ├── cross_city.py
    └── cold_start.py
```

---

# 29. 最终结论

Path-LLM 的核心并不是：

> “GPT-2 + 路径”。

而是：

\[
\boxed{
Domain Tokens
+
Cross-Modal Alignment
+
Adaptive Fusion
+
Pretrained Sequence Backbone
}
\]

迁移到 Next POI 后，最合理的版本不是简单复制：

```text
Topology + Text
```

而应该变成：

\[
\boxed{
POI Transition Topology
+
POI Semantics
+
Temporal Dynamics
}
\]

并形成：

```text
Static POI Knowledge
        ↓
Topology-Semantic Alignment
        ↓
Dynamic Context-Aware Fusion
        ↓
GPT-2 Mobility Modeling
        ↓
Candidate POI Matching
```

我目前最推荐的技术路线是：

> **以 Foursquare NYC/Tokyo 为主数据集，先复现 Path-LLM 式 Topology-Semantics 双模态框架，再加入 Temporal Dynamics，最终把创新点集中在动态融合和 discrepancy-aware curriculum learning 上。**

这条路线既保留了 Path-LLM 最值得借鉴的思想，也能避免论文变成简单的任务迁移。
