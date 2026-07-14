# software-poi-paper — 论文实验副本

本仓库是 [software-poi](https://github.com/your-org/software-poi) 论文实验的独立复现副本。包含论文《基于多模态数据融合的下一兴趣点推荐》的全部主实验、基线对比、消融实验及分析脚本，使用闭世界全排序评估协议。

## 与 `software-poi/` 的关系

`software-poi-paper` 从 `software-poi` 分叉而来，专为论文实验的**可复现性和可追溯性**而设计：

| 方面           | `software-poi/`（开发版）              | `software-poi-paper/`（论文版）                                                 |
| -------------- | ---------------------------------------- | --------------------------------------------------------------------------------- |
| **目的** | 功能开发、参数调优、探索实验             | 锁定论文最终配置，确保可复现                                                      |
| **配置** | 多种实验配置（optim、debug、profile 等） | 仅保留论文最终版`paper_nyc.yaml`、`paper_nyc_full.yaml` 和 `paper_tky.yaml` |
| **数据** | 引用外部数据                             | 自带`data/` 原始签到数据集副本                                                  |
| **结果** | 实验中间产物                             | 严格归档的`runs/paper_revision/`，含多种子、基线、消融                          |
| **脚本** | 通用训练/评估                            | 额外包含`summarize_paper_results.py` 等结果汇总工具                             |

---

## 1. 问题定义

给定 POI 集合 $\mathcal{V} = \{v_1, v_2, \dots, v_{|\mathcal{V}|}\}$ 和用户 $u$ 的历史签到序列 $S_u = [s_1, s_2, \dots, s_t]$（其中 $s_i = (v_i, \text{lat}_i, \text{lng}_i, c_i, \tau_i)$，$c_i$ 为类别，$\tau_i$ 为时间戳），**下一兴趣点推荐**任务的目标是预测用户下一个要访问的 POI：

$$
s_{t+1} = \arg\max_{v \in \mathcal{V}_{\text{cand}}} P(v \mid u, S_u)
$$

其中 $\mathcal{V}_{\text{cand}}$ 在闭世界协议下被限制为训练集中出现的 POI。

**核心挑战**：

- **稀疏性**：用户签到数据高度稀疏（每人平均签到次数有限）
- **多模态**：POI 包含拓扑、语义、空间、时间和用户偏好等多模态信息
- **统计规律**：用户的签到行为受到位置转移、时间周期和流行度等多重统计规律的影响

---

## 2. 贡献

本文的主要贡献如下：

1. **多模态 POI 表示学习框架**：将 Path-LLM 风格的多模态对齐（SemAlign）和门控融合（ContextFusion）引入 POI 推荐领域，融合拓扑、语义、空间、时间四模态信息 + 用户画像，并采用 GPT-2 序列骨干（pathllm_selective 微调策略）进行序列建模。
2. **统计先验融合机制**：设计 PriorEncoder（可学习的密集先验组合器）和稀疏统计先验体系（转移先验、历史转移、用户-POI、共访、历史重复等五种核心先验），使神经分数与统计分数线性叠加，显著提升推荐性能。
3. **端到端联合训练的可行性验证**：证明 PriorEncoder + 统计先验可以在完整模型中与神经骨干同时训练（end-to-end），效果与两阶段方法（先训神经骨干再微调 PriorEncoder）相当，简化了训练流程。
4. **系统性大规模实验**：在 Foursquare NYC 和 Tokyo 两个真实签到数据集上进行全排序闭世界评估，包含模态消融、组件消融、基线对比等多种消融实验，验证了方法的有效性（NYC 测试集 NDCG@10 = 30.20%, HR@10 = 45.95%）。

---

## 3. 模型结构

Path-LLM 风格的多模态 POI 表示学习框架 + 统计先验融合：

```
原始签到序列 / POI 元数据
    │
    ├── 拓扑模态：训练集 POI 转移图 + 转移统计 + Node2Vec
    ├── 语义模态：POI 文本描述 + 类别嵌入 + sentence-transformer 嵌入
    ├── 空间模态：归一化 GPS 坐标
    ├── 时间模态：本地小时、星期、签到间隔桶
    └── 用户画像：用户-类别统计先验 → 可学习画像投影
    │
    ▼
SemAlign：拓扑—语义跨模态对齐（InfoNCE 损失，instance-level + feature-level）
    │
    ▼
ContextFusion：上下文感知动态门控融合（sigmoid 门控加权拓扑+语义，再拼接空间+时间+用户+画像）
    │
    ▼
GPT-2 序列骨干（pathllm_selective 微调策略：冻结 MHA+FFN，仅训练 WPE+LN）
    │
    ▼
查询向量—候选 POI 向量归一化匹配（L2 normalize + logit_scale 缩放）
    │
    ▼
┌─────────────────────────────────────────────────────┐
│               统计先验融合层                          │
│                                                     │
│  (A) PriorEncoder（可学习）:                         │
│      4维密集先验 → Linear(4→1) → 学习组合权重        │
│                                                     │
│  (B) 稀疏先验（线性加权）:                            │
│      ├── 转移先验 (transition, weight=8.0)           │
│      ├── 历史转移 (history_transition, weight=4.0)   │
│      ├── 用户-POI (user_poi, weight=4.0)            │
│      ├── 共访先验 (co_visit, weight=0.5)             │
│      └── 历史重复 (history_repeat, weight=2.0)       │
│                                                     │
│  (C) CandidateReranker（可选）:                       │
│      10维特征 MLP 重排序                              │
│                                                     │
│  最终分数 = neural_scores + prior_scores              │
│  闭世界过滤：masked_fill(未见POI, -1e9)               │
└─────────────────────────────────────────────────────┘
```

模型1：

原始签到序列 / POI 元数据
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
┌────────────────────────────┐          ┌──────────────────────────────┐
│ Dual-scale Statistical     │          │ Multimodal POI Encoder       │
│ Profiler                   │          │                              │
│                            │          │ 拓扑模态                     │
│ ① 长期用户画像             │          │ 转移图 + Node2Vec/GNN        │
│   category distribution    │          │                              │
│   POI preference           │          │ 语义模态                     │
│   temporal preference      │          │ POI文本 + 类别语义           │
│   spatial preference       │          │ Sentence Transformer         │
│   repeat tendency          │          │                              │
│                            │          │ 空间模态                     │
│ ② 短期移动状态             │          │ GPS / 距离 / 区域编码        │
│   recent transition        │          │                              │
│   recent category          │          │ 时间模态                     │
│   current temporal pattern │          │ Hour / Weekday / Time Gap    │
│   recent spatial region    │          │                              │
└─────────────┬──────────────┘          └──────────────┬───────────────┘
              │                                         │
              │                          ┌──────────────▼──────────────┐
              │                          │ SemAlign                    │
              │                          │                            │
              │                          │ Topology ↔ Semantic        │
              │                          │ Instance-level InfoNCE     │
              │                          │ Feature-level Alignment    │
              │                          └──────────────┬──────────────┘
              │                                         │
              └──────────────────┬──────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────┐
              │ Prior-Conditioned Context Fusion         │
              │                                          │
              │ 长期画像 + 短期状态作为条件               │
              │ 动态控制四种模态的重要性                  │
              │                                          │
              │ α = Softmax(Gate(modalities, profiles))  │
              │                                          │
              │ fused POI sequence                       │
              └──────────────────┬───────────────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
       ┌───────────────────────┐     ┌─────────────────────────────┐
       │ Fused Check-in        │     │ Profile-Guided Candidate    │
       │ Sequence              │     │ Forecaster                  │
       │                       │     │                             │
       │ z1, z2, ..., zL       │     │ 多路召回：                  │
       └───────────┬───────────┘     │ · Transition neighbors      │
                   │                 │ · Semantic retrieval        │
                   │                 │ · User-history              │
                   │                 │ · Co-visit retrieval        │
                   │                 │ · Geographic retrieval      │
                   │                 │                             │
                   │                 │ 可学习候选精排              │
                   │                 │ Long-term candidates        │
                   │                 │ Short-term candidates       │
                   │                 └──────────────┬──────────────┘
                   │                                │
                   │                       Candidate summary tokens
                   │                                │
                   └────────────────┬───────────────┘
                                    ▼
              ┌───────────────────────────────────────────┐
              │ GPT-2 Sequence Backbone                  │
              │                                           │
              │ Input:                                    │
              │ [Long Profile Token]                      │
              │ [Short Mobility Token]                    │
              │ [Long Candidate Token]                    │
              │ [Short Candidate Token]                   │
              │ [z1, z2, ..., zL]                         │
              │                                           │
              │ pathllm_selective fine-tuning             │
              └───────────────────┬───────────────────────┘
                                  ▼
                         Next-POI Query Vector
                                  │
                                  ▼
              ┌───────────────────────────────────────────┐
              │ Candidate POI Neural Matching             │
              │                                           │
              │ score(p) = cosine(q, ep) / τ              │
              │                                           │
              │ Search within refined candidate set       │
              │                                           │
              │ 无固定权重统计先验加分                    │
              └───────────────────┬───────────────────────┘
                                  ▼
                         Next POI Recommendation


模型2：
                         原始签到数据
                Historical + Recent Trajectory
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
   多模态 POI 表征分支                多尺度行为先验分支
             │                                 │
   ┌─────────┼──────────┐           ┌──────────┴──────────┐
   │         │          │           │                     │
拓扑模态   语义模态   空间模态    长期用户画像           短期移动状态
   │         │          │           │                     │
Transition  Text       GPS/Grid   category/time/      transition/repeat/
Graph       Encoder    Relative   space/POI dist.     interval/co-visit
   │         │          │           │                     │
   └────┬────┘          │           ▼                     ▼
        │               │       Long-term              Short-term
        ▼               │       Prior Encoder          Prior Encoder
     SemAlign            │           │                     │
 拓扑—语义跨模态对齐      │           └──────────┬──────────┘
        │               │                      │
        └───────┬───────┘                      │
                │                              │
                └──────────────┬───────────────┘
                               ▼
                    Prior-Conditioned Fusion
                  先验条件化的签到 Token 构造
                               │
                               │
                  ┌────────────┴─────────────┐
                  │                          │
                  ▼                          ▼
          签到 Token 序列               动态候选生成
          z1,z2,...,zt              Multi-source Retrieval
                  │                          │
                  │                    Candidate Set
                  │                          │
                  │                    Candidate Memory
                  │                          │
                  └────────────┬─────────────┘
                               ▼
                [Long Profile Token]
                [Short Mobility Token]
                [Candidate Memory Token]
                [z1][z2]...[zt]
                               │
                               ▼
                    GPT-2 Sequence Backbone
                       + Lightweight LoRA
                               │
                               ▼
                         Query Vector
                               │
                               ▼
                  与 POI 多模态向量匹配
                               │
                               ▼
                       Top-K POI Prediction




### 3.1 统计先验数学定义

以下为核心统计先验的具体计算方式，$u$ 表示用户，$v$ 表示候选 POI，$S_u$ 为用户 $u$ 的历史序列：

| 先验     | 公式                                                                      | 含义                                                                           |
| -------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 转移先验 | $p_{\text{trans}}(v) = T(v_{t}, v)$                                     | 从上一 POI$v_t$ 到候选 $v$ 的训练转移概率                                  |
| 历史转移 | $p_{\text{ht}}(v) = \max_{k} \gamma^k \cdot T(s_{t-k}, v)$              | 历史中所有 POI 到候选的转移概率，$k$ 步前的转移乘以衰减系数 $\gamma = 0.8$ |
| 用户-POI | $p_{\text{up}}(v) = \log(\text{count}(u, v) + 1)$                       | 用户$u$ 访问候选 $v$ 的历史次数对数                                        |
| 共访先验 | $p_{\text{cv}}(v) = \log(\sum_{v' \in S_u} \text{co_count}(v', v) + 1)$ | 用户访问过的 POI 与候选$v$ 的共访次数对数                                    |
| 历史重复 | $p_{\text{hr}}(v) = \max_{k} \delta^k \cdot \mathbb{1}(s_{t-k} = v)$    | 候选$v$ 在历史中出现的位置加权，衰减系数 $\delta = 0.7$                    |

其中 $T$ 为训练集构建的 POI→POI 转移矩阵，$\gamma, \delta$ 为衰减系数。PriorEncoder 对 4 种密集先验（流行度、类别转移、用户类别、空间）学习非线性组合 $f_{\text{PE}}(\mathbf{p}) = \mathbf{W}\mathbf{p}$，其中 $\mathbf{W} \in \mathbb{R}^{1 \times 4}$。

### 3.2 PriorEncoder（可学习先验融合）

PriorEncoder 输入 4 维密集先验特征（流行度、类别转移概率、用户类别偏好、空间距离分数），通过学习一个线性层 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ 得到最终先验分数。该设计避免了全连接层在大候选集上的 OOM 问题（候选 POI 数量 ~34k），同时保留了可学习的先验组合能力。

### 3.3 训练与评估阶段先验策略差异

训练时（`apply_priors_during_training: false`），稀疏统计先验的权重全部设为 0，仅 PriorEncoder 通过 4 维密集先验学习组合权重。评估时（`include_priors=True`），所有稀疏先验按配置权重同时参与计算。这种设计权衡了以下因素：

- **训练效率**：稀疏先验需对每个候选 POI 循环计算，在大候选集下会显著降低训练速度
- **学习目标一致性**：PriorEncoder 可学习到合理的先验组合，评估时加入稀疏先验后整体表现仍最优
- **两阶段兼容性**：支持先训骨干 → 微调 PriorEncoder 的灵活流程

---

## 4. 数据集与实验协议

### 4.1 数据来源

- **Source**: TSMC2014/Foursquare NYC 和 Tokyo 签到数据
- **NYC**: `data/dataset_TSMC2014_NYC.txt` → `processed/NYC/`
- **TKY**: `data/dataset_TSMC2014_TKY.txt` → `processed/TKY_paper/`

### 4.2 切分协议

遵循 `protocol.md`：

1. 每个用户按 UTC 时间排序
2. 使用时区偏移计算本地小时和星期
3. 过滤签到次数 < 2 的用户
4. 按用户时间顺序 80%/10%/10% 切分训练/验证/测试
5. 历史长度上限 `max_seq_len=20`
6. 所有行为统计（转移图、Node2Vec、流行度等）仅由训练前缀构建
7. **闭世界评估**：候选和评估目标限制在训练集中出现过的 POI

### 4.3 数据集统计

| 项目         |       NYC       |       TKY       |
| ------------ | :-------------: | :-------------: |
| 用户数       |      1,083      |      2,293      |
| POI 数       |     38,333     |     61,858     |
| 类别数       |       400       |       385       |
| 训练样本     |     180,873     |     456,714     |
| 验证样本     |     22,736     |     57,348     |
| 测试样本     |     22,736     |     57,348     |
| 训练可见 POI |     34,172     |     55,017     |
| 转移边       |     133,305     |     269,599     |
| 平均坐标     | (40.75, -73.97) | (35.68, 139.71) |

### 4.4 模型参数量与计算成本

| 项目                                     |               数值               |
| ---------------------------------------- | :-------------------------------: |
| 模型总参数量                             |              ≈690K              |
| GPT-2 骨干                               |      2 层、4 头、128 维嵌入      |
| 每 epoch 训练时间（NYC, batch_size=512） |        ~3 分钟（RTX 4090）        |
| 完整训练（3 epoch）                      |             ~10 分钟             |
| 对齐预训练（1 epoch）                    |              ~1 分钟              |
| GPU 显存需求                             | <2 GB（batch_size=512, AMP BF16） |

---

## 5. 环境安装

```bash
pip install -e .
```

或手动安装：

```bash
pip install torch transformers pandas numpy scikit-learn networkx node2vec gensim pyyaml tqdm safetensors tokenizers
```

需要加载 HuggingFace 模型：

- `gpt2`
- `sentence-transformers/all-MiniLM-L6-v2`

---

## 6. 快速运行

### 预处理

```bash
python scripts/preprocess.py --config configs/paper_nyc.yaml   # NYC
python scripts/preprocess.py --config configs/paper_tky.yaml   # TKY
```

### 对齐预训练

```bash
python scripts/pretrain_alignment.py --config configs/paper_nyc.yaml
```

### 训练

#### 纯神经骨干（消融对照）

```bash
python scripts/train.py --config configs/paper_nyc.yaml
```

#### 完整端到端模型（神经骨干 + PriorEncoder + 统计先验）

```bash
python scripts/train.py --config configs/paper_nyc_full.yaml
```

### 评估

```bash
python scripts/evaluate.py --checkpoint runs/paper_revision/NYC/full_seed42_end2end/best.pt --split test
```

### 消融实验

```bash
# 模态消融（基于 paper_nyc_full.yaml 修改对应 flag）
python scripts/train.py --config configs/paper_nyc_no_topo.yaml     # 禁用拓扑
python scripts/train.py --config configs/paper_nyc_no_sem.yaml     # 禁用语义
python scripts/train.py --config configs/paper_nyc_no_spatial.yaml   # 禁用空间
python scripts/train.py --config configs/paper_nyc_no_temporal.yaml  # 禁用时间
python scripts/train.py --config configs/paper_nyc_no_user.yaml    # 禁用用户ID嵌入

# 组件消融
python scripts/train.py --config configs/paper_nyc_no_align.yaml   # 禁用跨模态对齐
python scripts/train.py --config configs/paper_nyc_no_gate.yaml    # 门控→简单拼接
python scripts/train.py --config configs/paper_nyc_no_priorencoder.yaml  # 禁用 PriorEncoder
```

### 基线实验

```bash
python scripts/run_baseline.py --config configs/paper_nyc.yaml --model gru4rec
python scripts/run_baseline.py --config configs/paper_nyc.yaml --model sasrec
python scripts/run_baseline.py --config configs/paper_nyc.yaml --model bert4rec
python scripts/run_baseline.py --config configs/paper_nyc.yaml --model mostpopular
python scripts/run_baseline.py --config configs/paper_nyc.yaml --model markov
```

### 结果汇总

```bash
python scripts/summarize_paper_results.py --root runs/paper_revision
```

---

## 7. 实验结果

### 7.1 NYC 测试集 — 完整端到端模型结果（seed 42, 3 epoch）

对应 `configs/paper_nyc_full.yaml`（PriorEncoder + 统计先验，端到端 3 epochs 训练）：

|  指标  | 验证集（最佳 epoch 2） | 测试集 |
| :-----: | :--------------------: | :----: |
|  HR@5  |         38.98%         | 37.71% |
|  HR@10  |         48.16%         | 45.95% |
| NDCG@5 |         28.12%         | 27.52% |
| NDCG@10 |         31.11%         | 30.20% |
|  HR@20  |         55.35%         | 52.76% |
|   MRR   |         26.58%         | 26.04% |

### 7.2 NYC — 消融实验（1 epoch, seed 42）

基于 `configs/paper_nyc_full.yaml` 修改对应参数的消融实验（统一 1 epoch, seed 42, batch_size=512），测试集评估结果：

| 消融变体                      | 修改                             |  HR@5  | HR@10 | NDCG@10 |  MRR  |
| ----------------------------- | -------------------------------- | :----: | :----: | :-----: | :----: |
| **完整模型（3 epoch）** | —                               | 37.71% | 45.95% | 30.20% | 26.04% |
| w/o Topology                  | `use_topology_modality: false` | 37.30% | 44.54% | 29.91% | 25.97% |
| w/o Semantic                  | `use_semantic_modality: false` | 36.82% | 44.93% | 29.83% | 25.85% |
| w/o Spatial                   | `use_spatial_modality: false`  | 37.15% | 45.46% | 29.95% | 25.83% |
| w/o Temporal                  | `use_temporal_modality: false` | 37.52% | 45.68% | 30.21% | 26.12% |
| w/o User ID                   | `use_user_id_embedding: false` | 37.20% | 45.35% | 30.04% | 26.00% |
| w/o SemAlign                  | `align_loss_weight: 0`         | 37.59% | 45.84% | 30.31% | 26.20% |
| w/o Gate                      | `fusion_mode: concat`          | 38.03% | 46.17% | 30.57% | 26.42% |
| w/o PriorEncoder              | `use_prior_encoder: false`     |   —   |   —   |   —   |   —   |

> 注：w/o SemAlign 评估中，w/o PriorEncoder（`apply_priors_during_training: true` 导致训练缓慢）待补充。完整模型 3 epoch 结果作为基准。消融实验均为 1 epoch。

### 7.3 NYC — 结果对比

| 模型                                   |       HR@5       |      HR@10      |     NDCG@10     |       MRR       |
| -------------------------------------- | :--------------: | :--------------: | :--------------: | :--------------: |
| **MostPopular**                  |      2.11%      |      3.12%      |      1.60%      |      1.40%      |
| **Markov**                       |      22.46%      |      24.92%      |      18.45%      |      16.70%      |
| **GRU4Rec**                      |      21.41%      |      24.07%      |      18.05%      |      16.51%      |
| **SASRec**                       |      16.87%      |      19.07%      |      14.24%      |      13.05%      |
| **BERT4Rec**                     |      17.14%      |      19.32%      |      14.42%      |      13.22%      |
| **纯神经骨干 (seed 42)**         |      30.60%      |      38.16%      |      23.96%      |      20.28%      |
| **Pure Neural (3种子均值)**      |      30.88%      |      38.44%      |      24.16%      |      20.47%      |
| **PriorEncoder微调 (seed 42)**   |      37.52%      |      46.63%      |      30.09%      |      25.74%      |
| **PriorEncoder微调 (2种子均值)** |      37.74%      |      46.86%      |      30.25%      |      25.88%      |
| **完整端到端 (seed 42, ours)**   | **37.71%** | **45.95%** | **30.20%** | **26.04%** |

> 完整端到端训练将 Path-LLM 风格的多模态神经骨干与统计先验融合在 3 个 epoch 中同时训练，效果与两阶段方法（先训神经骨干再微调 PriorEncoder）相当，证明了端到端训练的可行性和有效性。

### 7.4 NYC — 纯神经骨干 3 种子复现结果（消融对照）

对应 `configs/paper_nyc.yaml`（所有 prior weight=0，`use_prior_encoder=false`）：

|      Seed      |       HR@5       |      HR@10      |     NDCG@10     |       MRR       |
| :------------: | :--------------: | :--------------: | :--------------: | :--------------: |
|       42       |      30.60%      |      38.16%      |      23.96%      |      20.28%      |
|      2025      |      30.92%      |      38.52%      |      24.26%      |      20.57%      |
|      2026      |      31.12%      |      38.63%      |      24.26%      |      20.56%      |
| **均值** | **30.88%** | **38.44%** | **24.16%** | **20.47%** |

### 7.5 NYC — 先验消融实验（基于 PriorEncoder seed42 checkpoints）

在先验完整模型（PriorEncoder）的 best checkpoints 上评估不同先验组合的贡献：

| 消融变体    | 启用的先验                |  HR@5  | HR@10 | NDCG@10 |  MRR  |
| ----------- | ------------------------- | :----: | :----: | :-----: | :----: |
| Full        | 全部（PriorEncoder 学习） | 37.52% | 46.63% | 30.09% | 25.74% |
| Transition  | 转移 + 历史转移           | 37.52% | 46.63% | 30.09% | 25.74% |
| User/Repeat | 用户-POI + 共访 + 重复    | 33.11% | 43.37% | 25.77% | 21.24% |
| Pop/Spatial | 流行度 + 空间             | 1.42% | 2.48% |  1.18%  | 1.30% |
| None        | 无统计先验                | 1.42% | 2.48% |  1.18%  | 1.30% |

> 注：PriorEncoder 在训练时学到了内部权重分配。仅用转移先验即可恢复完整模型性能，说明转移信号是先验中最关键的。

### 7.6 TKY 数据集实验结果

#### 7.6.1 TKY — PriorEncoder（3 epoch）

对应 `configs/paper_tky_prior.yaml`（PriorEncoder + 统计先验）：

| Seed |  HR@5  | HR@10 | NDCG@5 | NDCG@10 |  MRR  |
| :--: | :----: | :----: | :----: | :-----: | :----: |
|  42  | 40.46% | 50.66% | 28.64% | 31.95% | 27.03% |

#### 7.6.2 TKY — 消融实验（正在运行中）

TKY 完整模型 1 epoch + 模态消融 x5 + 组件消融 x3 正在后台训练中，完成后会更新结果。

---

## 8. 相关工作与讨论

### 8.1 与现有 POI 推荐方法对比

本文方法与其他 POI 推荐领域代表性方法的对比定位：

| 方法               |    序列建模    |          多模态          | 统计先验 | 与本文关系                      |
| ------------------ | :------------: | :----------------------: | :------: | ------------------------------- |
| **ST-RNN**   |      RNN      |        空间+时间        |    ✗    | 早期尝试，缺乏拓扑和语义        |
| **DeepMove** | RNN+Attention |        空间+时间        |    ✗    | 引入历史注意力，模态有限        |
| **STAN**     | Self-Attention |        空间+时间        |    ✗    | 基于 Transformer，仍缺语义      |
| **GETNext**  |      GNN      |        拓扑+时间        |    ✓    | 使用 GNN 和流行度，类似先验思路 |
| **Path-LLM** |     GPT-2     |        拓扑+语义        |    ✗    | 本文骨干来源，无 POI 统计先验   |
| **本文方法** |     GPT-2     | 拓扑+语义+空间+时间+用户 |    ✓    | 综合多模态 + 统计先验           |

### 8.2 损失函数

模型使用两种损失函数的加权组合：

1. **推荐损失（Cross-Entropy）**：

   $$
   \mathcal{L}_{\text{rec}} = -\frac{1}{B}\sum_{i=1}^B \log\frac{\exp(s(v_{\text{target}}))}{\sum_{v \in \mathcal{V}_{\text{cand}}}\exp(s(v))}
   $$

   其中 $s(v)$ 为候选 POI $v$ 的最终分数。
2. **跨模态对齐损失**（InfoNCE, instance-level + feature-level）：

   $$
   \mathcal{L}_{\text{align}} = \mathcal{L}_{\text{instance}} + \lambda_f \cdot \mathcal{L}_{\text{feature}}
   $$

   其中 $\mathcal{L}_{\text{instance}}$ 为拓扑-语义同序列实例级对比损失，$\mathcal{L}_{\text{feature}}$ 为特征级对比损失。总损失：

   $$
   \mathcal{L} = \mathcal{L}_{\text{rec}} + \alpha \cdot \mathcal{L}_{\text{align}}
   $$

   配置参数：$\alpha = 0.05$（`align_loss_weight`），$\lambda_f = 0.25$（`align_feature_weight`）。

---

## 9. 配置文件

| 配置                                       | 用途                                                       |
| ------------------------------------------ | ---------------------------------------------------------- |
| `configs/paper_nyc.yaml`                 | NYC 纯神经骨干基线（batch_size=512, epochs=3, 无统计先验） |
| `configs/paper_nyc_full.yaml`            | NYC 完整端到端模型（PriorEncoder + 全统计先验，3 epochs）  |
| `configs/paper_nyc_no_topo.yaml`         | 消融：禁用拓扑模态                                         |
| `configs/paper_nyc_no_sem.yaml`          | 消融：禁用语义模态                                         |
| `configs/paper_nyc_no_spatial.yaml`      | 消融：禁用空间模态                                         |
| `configs/paper_nyc_no_temporal.yaml`     | 消融：禁用时间模态                                         |
| `configs/paper_nyc_no_user.yaml`         | 消融：禁用用户 ID 嵌入                                     |
| `configs/paper_nyc_no_align.yaml`        | 消融：禁用跨模态对齐（SemAlign）                           |
| `configs/paper_nyc_no_gate.yaml`         | 消融：门控→简单拼接                                       |
| `configs/paper_nyc_no_priorencoder.yaml` | 消融：禁用 PriorEncoder                                    |
| `configs/paper_tky.yaml`                 | TKY 纯神经骨干配置                                         |
| `configs/paper_tky_prior.yaml`           | TKY PriorEncoder 配置（3 epoch）                           |
| `configs/paper_tky_full_1epoch.yaml`     | TKY 完整模型 1 epoch                                       |
| `configs/paper_tky_no_*.yaml`            | TKY 对应消融配置（8个）                                    |
| `configs/debug.yaml`                     | 小规模调试配置                                             |

### 主实验关键参数（`paper_nyc_full.yaml`）

| 参数                             |         数值         | 说明                                        |
| -------------------------------- | :-------------------: | ------------------------------------------- |
| `hidden_dim`                   |          128          | 公共隐藏维度                                |
| `node2vec_dim`                 |          128          | Node2Vec 嵌入维度                           |
| `batch_size`                   |          512          | 批次大小                                    |
| `epochs`                       |           3           | 训练轮数                                    |
| `lr`                           |        0.0005        | 学习率                                      |
| `semantic_encoder`             |  `hf_transformer`  | 语义编码器类型                              |
| `text_embedding_dim`           |          384          | 文本嵌入维度                                |
| `candidate_protocol`           |   `closed_world`   | 闭世界评估                                  |
| `monitor_metric`               |      `NDCG@10`      | 验证监控指标                                |
| `gpt_freeze_policy`            | `pathllm_selective` | GPT-2 微调策略（冻结 MHA+FFN，训练 WPE+LN） |
| `fusion_mode`                  |   `context_gate`   | 动态门控融合                                |
| `use_prior_encoder`            |       `true`       | 启用 PriorEncoder                           |
| `apply_priors_during_training` |       `false`       | 训练时仅 PriorEncoder，评估时全部先验生效   |
| `align_loss_weight`            |         0.05         | TPalign 对齐损失权重                        |

### 统计先验权重（`paper_nyc_full.yaml`）

| 先验                     | 权重 | 模式                 |
| ------------------------ | :--: | -------------------- |
| transition_prior         | 8.0 | prob                 |
| history_transition_prior | 4.0 | prob, decay=0.8, max |
| user_poi_prior           | 4.0 | log_count            |
| co_visit_prior           | 0.5 | log_count            |
| history_repeat_prior     | 2.0 | decay=0.7, max       |

---

## 10. 目录结构

```
software-poi-paper/
├── configs/              # 论文实验 YAML 配置
│   ├── paper_nyc*.yaml         # NYC 主实验 + 消融配置
│   ├── paper_tky*.yaml         # TKY 主实验 + 消融配置
│   └── ...
├── data/                 # 原始签到数据集（TSMC2014）
├── poi_rec/              # 核心代码
│   ├── data/             # 数据处理（预处理、数据集、候选生成）
│   ├── losses/           # 损失函数（对齐损失、推荐损失）
│   ├── models/           # 模型（编码器、对齐、融合、GPT 骨干、先验）
│   ├── training/         # 训练/评估/指标/检查点
│   └── utils/            # 配置加载、随机种子
├── processed/            # 预处理后的数据
│   ├── NYC/              # NYC 处理后数据
│   └── TKY_paper/        # TKY 处理后数据
├── runs/                 # 实验结果归档
│   └── paper_revision/   # 论文修订版实验
│       ├── logs/         # 训练日志
│       ├── NYC/          # NYC 主实验 + 消融 + 基线
│       │   ├── full_seed42_end2end/    # 完整端到端模型结果 ✅
│       │   ├── ablations/              # 模态+组件消融 (7/8 ✅)
│       │   ├── baselines/              # 基线实验结果
│       │   └── prior_ablation/         # 先验消融
│       └── TKY/          # TKY 实验
│           ├── prior_seed42/           # TKY PriorEncoder seed42 ✅
│           └── ablations/              # TKY 消融 (正在运行)
└── scripts/              # 命令行入口脚本
    ├── train.py          # 训练
    ├── evaluate.py       # 评估
    ├── preprocess.py     # 预处理
    ├── pretrain_alignment.py  # 对齐预训练
    ├── run_baseline.py   # 基线模型运行
    ├── paired_bootstrap.py  # 统计显著性检验（bootstrap）
    ├── run_ablation_suite.sh  # 消融实验脚本
    └── summarize_paper_results.py  # 结果汇总
```

### 核心模型文件映射

| 文件                                      | 功能                                                    |
| ----------------------------------------- | ------------------------------------------------------- |
| `poi_rec/models/poi_model.py`           | 总模型、候选打分、统计先验融合                          |
| `poi_rec/models/encoders.py`            | 5 种编码器（拓扑、语义、空间、时间、用户画像）          |
| `poi_rec/models/alignment.py`           | SemAlign 投影头（对应 Path-LLM TPalign）                |
| `poi_rec/models/fusion.py`              | ContextFusion / DynamicFusion（对应 Path-LLM TPfusion） |
| `poi_rec/models/gpt_backbone.py`        | GPT-2 序列骨干                                          |
| `poi_rec/models/priors.py`              | PriorEncoder 和 CandidateReranker                       |
| `poi_rec/losses/alignment_loss.py`      | InfoNCE 对齐损失（instance-level + feature-level）      |
| `poi_rec/losses/recommendation_loss.py` | 推荐损失（Cross-Entropy）                               |
| `poi_rec/data/candidates.py`            | 动态候选生成器                                          |

---

## 11. 局限性与未来工作

1. **POI 领域专用基线缺失**：当前基线为通用序列推荐方法（GRU4Rec/SASRec/BERT4Rec）和传统方法（Markov/MostPopular），建议补充 ST-RNN、DeepMove、STAN、GETNext 等 POI 领域 SOTA 对比。
2. **TKY 消融实验进行中**：TKY 数据集消融（模态消融 + 组件消融 + 1 epoch 完整模型）正在后台训练中。
3. **统计显著性分析**：`scripts/paired_bootstrap.py` 可用于计算 checkpoints 间的 bootstrap 置信区间，结果可增强实验结论的可信度。
4. **TOCL 课程学习**：Path-LLM 的两阶段重叠课程学习（TOCL）仅部分实现，完全实现后可能进一步提升性能。
5. **超参数敏感性分析**：`align_loss_weight`、先验权重、`hidden_dim` 等关键超参数的敏感性分析有助于理解模型行为。

---

## 12. 文件说明

- `protocol.md` — 详细实验协议（数据切分、候选协议、指标定义）
- `pyproject.toml` — 项目元数据和依赖声明
- `tasks_current.md` — 模型架构分析文档（含与 Path-LLM 的详细对比）
- `run_all_ablations.sh` — 一键运行全部消融实验的脚本
- 单元测试：`tests/` 目录下

---

## 参考

- 实验协议：`protocol.md`
- 开发版主仓库：`../software-poi/`
- 论文源码和 LaTeX：`../software-poi/latex/`
