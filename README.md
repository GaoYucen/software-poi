# 基于多模态数据融合的下一兴趣点推荐

本仓库实现了论文《基于多模态数据融合的下一兴趣点推荐方法》中的核心实验代码。任务目标是根据用户历史签到序列预测下一次访问的 POI（Point-of-Interest）。当前实现面向 TSMC2014/Foursquare 签到数据，主实验使用 NYC 数据集，并采用严格的时间顺序切分、训练集构图和闭世界全排序评估协议。

代码对应一个 Path-LLM 风格的多模态 POI 表示学习框架：先从签到日志和 POI 元数据构建拓扑、语义、空间、时间等模态，再通过 SemAlign 完成拓扑—语义对齐，通过 ContextFusion 进行上下文感知融合，最后使用选择性微调的 GPT-2 骨干网络完成序列建模和下一 POI 匹配。

## 1. 模型结构

整体流程可概括为：

```text
原始签到序列 / POI元数据
        │
        ├── 拓扑模态：训练集 POI 转移图 + 转移统计 + Node2Vec
        ├── 语义模态：POI 文本描述 + 类别嵌入 + Transformer 文本嵌入
        ├── 空间模态：归一化 GPS 坐标
        └── 时间模态：本地小时、星期、签到间隔桶
        │
        ▼
SemAlign：拓扑—语义跨模态对齐
        │
        ▼
ContextFusion：上下文感知动态门控融合
        │
        ▼
选择性微调 GPT-2 序列骨干
        │
        ▼
查询向量—候选 POI 向量归一化匹配
        │
        ▼
下一 POI 全排序预测
```

### 1.1 多模态编码器

代码入口主要位于：`poi_rec/models/poi_model.py`、`poi_rec/models/encoders.py`、`poi_rec/models/alignment.py`、`poi_rec/models/fusion.py`、`poi_rec/models/gpt_backbone.py`。

| 模态 | 输入来源 | 实现方式 |
| --- | --- | --- |
| 拓扑模态 | 训练集相邻签到诱导的有向加权 POI 转移图 | 转移统计特征 + Node2Vec 嵌入，经线性投影和 LayerNorm 得到拓扑表示 |
| 语义模态 | POI 类别、城市和文本描述 | `sentence-transformers/all-MiniLM-L6-v2` 生成文本嵌入，叠加可学习类别嵌入 |
| 空间模态 | POI 经纬度 | 坐标归一化后经 MLP/GELU/LayerNorm 投影到隐藏空间 |
| 时间模态 | 签到时间上下文 | 本地小时、星期、时间间隔 bucket 分别嵌入后求和并归一化 |
| 用户/画像信息 | 训练集用户类别偏好统计、用户 ID | 用户 ID embedding 与用户类别画像投影作为辅助上下文 |

为避免信息泄漏，所有行为统计类特征（转移图、Node2Vec、用户-POI/用户-类别统计等）均只由训练集构建。

### 1.2 SemAlign 跨模态对齐

SemAlign 用于缓解拓扑表示和语义表示之间的模态鸿沟。实现中包含 `topology_projection` 和 `semantic_projection` 两个投影头。训练时结合实例级和特征级 InfoNCE 对齐损失：实例级对齐关注同一 POI 的拓扑视角和语义视角是否接近，特征级对齐鼓励两个模态在特征子空间上保持一致。

正式配置中：`align_loss_weight=0.05`、`align_instance_weight=1.0`、`align_feature_weight=0.25`、`alignment_pretrain_epochs=1`。

### 1.3 ContextFusion 上下文感知融合

`DynamicFusion` 对应论文中的 ContextFusion。它不是简单拼接或固定加权，而是根据当前序列上下文动态生成门控向量：

```text
gate = sigmoid(MLP([topology, semantic, temporal, user, profile]))
static = gate * topology + (1 - gate) * semantic
token = MLP([static, spatial, temporal, user, profile])
```

因此，模型可以在不同用户、不同轨迹位置和不同时间上下文下，自适应调整拓扑信息与语义信息的权重。

### 1.4 统计先验特征（Statistical Priors）

评估时，模型在神经匹配分数之外，可以叠加以下统计先验分数来提升推荐精度。这些先验全部由训练集统计生成，不会造成信息泄漏。

| 先验 | 配置参数 | 说明 |
| --- | --- | --- |
| POI 全局流行度 | `popularity_prior_weight` | 训练集 visit 次数取 log1p 归一化 |
| 最近转移 | `transition_prior_weight` | 上一步 POI → 候选 POI 的转移概率 |
| 历史多步转移 | `history_transition_prior_weight` | 历史中所有 POI 的转移概率，按距离衰减 |
| 用户-POI 偏好 | `user_poi_prior_weight` | 用户历史访问过的 POI |
| 用户类别偏好 | `user_category_prior_weight` | 用户对 POI 类别的偏好 |
| 类别转移 | `category_transition_prior_weight` | 上一步 POI 类别 → 候选 POI 类别的转移概率 |
| 重复访问 | `history_repeat_prior_weight` | 历史中出现过的 POI，按距离衰减 |
| 空间邻近 | `spatial_prior_weight` | 上一步 POI 与候选 POI 的 GPS 距离（高斯核） |
| 共访偏好 | `co_visit_prior_weight` | 与上一步 POI 被同一用户访问过的 POI |

最终的预测分数为：

```text
final_score = neural_score + Σ(weight_i * statistical_prior_i)
```

训练期可以通过 `apply_priors_during_training: false` 保持纯神经训练（速度快），而评估时模型默认自动包含所有启用的先验。

详细配置见 `configs/nyc_4090.yaml` 中的 `# --- Statistical priors ---` 节。

### 1.5 GPT-2 序列建模与候选匹配

融合后的 token 序列输入 GPT-2 骨干网络。正式 NYC 配置使用 `gpt2`、`hidden_dim=128`、`gpt_layers=2`、`gpt_heads=4`、`max_seq_len=20`，并采用 `gpt_freeze_policy: pathllm_selective`。该策略只更新 GPT-2 的位置嵌入和 LayerNorm 参数，冻结主要 attention/MLP 子层，以保留预训练序列建模能力并降低训练成本。

推理时，模型取最后一个有效位置的隐藏状态作为查询向量，并与全部候选 POI 表示做归一化点积匹配。主实验使用 `closed_world` 协议，即候选和评估目标限制在训练集中出现过的 POI。

## 2. 数据集情况

### 2.1 数据来源

数据来自 TSMC2014/Foursquare 签到数据，原始字段包括用户 ID、POI/venue ID、类别 ID 与类别名称、纬度、经度、时区偏移和 UTC 签到时间。

当前仓库包含 NYC 数据配置：

- 原始文件：`data/dataset_TSMC2014_NYC.txt`
- 处理后目录：`processed/NYC`
- 主实验配置：`configs/nyc_mvp.yaml`

### 2.2 预处理与切分协议

预处理遵循 `protocol.md` 中定义的实验协议：

1. 按用户对签到记录进行 UTC 时间排序；
2. 使用时区偏移计算本地小时和星期；
3. 过滤签到次数少于 2 的用户；
4. 按用户时间顺序进行比例切分：训练集 80%、验证集 10%、测试集 10%；
5. 每个目标位置构造一个下一 POI 样本，历史长度最多为 `max_seq_len=20`；
6. 所有拓扑、流行度和用户偏好统计仅由训练前缀生成；
7. 主实验采用闭世界全排序：只评估训练集中出现过的 POI 目标。

### 2.3 当前处理后数据统计

根据当前 `processed/NYC/metadata.json`，代码实际处理后的 NYC 数据规模如下：

| 项目 | 数值 |
| --- | ---: |
| 用户数 | 1,083 |
| POI 数 | 38,333 |
| 类别数 | 400 |
| 训练样本数 | 180,873 |
| 验证样本数 | 22,736 |
| 测试样本数 | 22,736 |
| 训练集可见 POI | 34,172 |
| 转移边数 | 133,305 |
| 训练相邻转移次数 | 180,873 |

论文正文中的数据统计表使用了另一个整理口径（NYC：1,007 用户、5,013 POI、210,810 签到、269 类别）。当前 README 以代码目录中最新 `processed/NYC/metadata.json` 为准；如需完全复现实验论文表格，应确认使用的原始数据过滤口径和预处理产物是否一致。

## 3. 代码运行和性能

### 3.1 环境依赖

建议使用已有 conda 环境 `py11`：

```bash
conda run -n py11 python -m pip install transformers safetensors tokenizers tqdm networkx
```

正式 NYC 配置需要能够加载 `gpt2` 和 `sentence-transformers/all-MiniLM-L6-v2`。如果无法联网下载 HuggingFace 模型，需要提前将模型缓存到本地。`fallback_to_random_gpt: true` 只建议用于 debug 或 smoke test，不建议用于正式实验。

### 3.2 Debug Smoke Test

```bash
conda run -n py11 python scripts/preprocess.py --config configs/debug.yaml
conda run -n py11 python scripts/pretrain_alignment.py --config configs/debug.yaml
conda run -n py11 python scripts/train.py --config configs/debug.yaml
conda run -n py11 python scripts/evaluate.py --checkpoint runs/debug/best.pt --split val
```

也可以用 `--override` 临时覆盖配置：

```bash
conda run -n py11 python scripts/train.py --config configs/debug.yaml --override epochs=5 --override run_dir=runs/debug_e5
conda run -n py11 python scripts/evaluate.py --checkpoint runs/debug/best.pt --split val --override transition_prior_weight=16
```

### 3.3 NYC 主实验运行

```bash
conda run -n py11 python scripts/preprocess.py --config configs/nyc_mvp.yaml
conda run -n py11 python scripts/pretrain_alignment.py --config configs/nyc_mvp.yaml
conda run -n py11 python scripts/train.py --config configs/nyc_mvp.yaml
conda run -n py11 python scripts/evaluate.py --checkpoint runs/nyc_mvp/best.pt --split test
```

主配置关键参数：

| 参数 | 数值 |
| --- | --- |
| `hidden_dim` | 128 |
| `node2vec_dim` | 128 |
| `batch_size` | 64 |
| `epochs` | 5 |
| `lr` | 0.0005 |
| `weight_decay` | 0.01 |
| `alignment_pretrain_epochs` | 1 |
| `candidate_protocol` | `closed_world` |
| `monitor_metric` | `NDCG@10` |

正式 `nyc_mvp` 配置禁用了统计先验和协同打分，因此论文报告的主结果主要反映纯神经多模态表示学习框架的能力，而不是依赖手工统计先验重排序。

### 3.4 Alignment Checkpoint 复用

`alignment_pretrain.pt` 只需生成一次，后续训练可以自动复用，节省预训练时间。

**复用条件**：checkpoint 中的 `preprocess_fingerprint`、`num_pois`、`num_categories`、`node2vec_dim`、`text_embedding_dim` 必须与当前预处理数据完全匹配。

配置示例（`configs/nyc_4090.yaml`）：

```yaml
alignment_checkpoint: runs/nyc_4090_epoch1/alignment_pretrain.pt
reuse_alignment_checkpoint: true
```

- `alignment_checkpoint`：显式指定 checkpoint 路径
- `reuse_alignment_checkpoint: true`：运行 `scripts/pretrain_alignment.py` 时，若存在兼容的 checkpoint 则直接跳过并 log 复用信息

同一预处理数据下，可以跨 run 目录复用同一个 checkpoint，无需每次重新生成。

### 3.5 NYC 4090 配置运行（含统计先验）

利用 RTX 4090 的更大 batch size 和 `bf16` 加速，1 轮训练即可快速评估统计先验效果：

```bash
conda run -n py11 python scripts/pretrain_alignment.py --config configs/nyc_4090.yaml
conda run -n py11 python scripts/train.py --config configs/nyc_4090.yaml
conda run -n py11 python scripts/evaluate.py --checkpoint runs/nyc_4090_epoch1/best.pt --split test
```

配置要点：

| 参数 | 数值 | 说明 |
| --- | --- | --- |
| `batch_size` | 512 | 4090 高吞吐 |
| `epochs` | 1 | 快速观察先验效果 |
| `apply_priors_during_training` | false | 训练期纯神经 |
| 9 种统计先验权重 | >0 | 评估期分数重排 |

1 轮训练（纯神经）+ 统计先验评估的测试集结果：

| 指标 | 论文 5 轮（无先验） | 1 轮 + 统计先验 |
| --- | ---: | ---: |
| HR@5 | 30.70% | **37.74%** |
| NDCG@5 | 22.01% | **27.62%** |
| HR@10 | 37.66% | **46.36%** |
| NDCG@10 | 24.28% | **30.43%** |
| HR@20 | 43.34% | **53.03%** |
| MRR | 20.80% | **26.21%** |

> 注意：此处目的是评估统计先验的增量收益。即使只训练 1 轮，加上统计先验后各指标均显著高于论文 5 轮纯神经结果，说明统计先验特征是 POI 推荐任务中高效且互补的信号。

### 3.6 测试

```bash
conda run -n py11 python -m unittest discover -s tests
```

### 3.7 当前性能结果

论文中报告的完整模型在 NYC 测试集上的主结果如下：

| 指标 | 验证集 | 测试集 |
| --- | ---: | ---: |
| HR@5 | 31.14% | 30.70% |
| NDCG@5 | 21.83% | 22.01% |
| HR@10 | 39.05% | 37.66% |
| NDCG@10 | 24.41% | 24.28% |
| HR@20 | 45.36% | 43.34% |
| MRR | 20.60% | 20.80% |

训练过程中，验证集最佳 epoch 为第 4 轮，训练损失从第 1 轮约 6.36 平滑下降到第 5 轮约 4.64。验证集与测试集的 NDCG@10 差距约 0.13 个百分点，说明当前训练和无泄漏切分下泛化较稳定。

#### 3.7.1 与基准方法对比

以下为论文中报告的各基准方法在 NYC 测试集上的完整性能对比。本文方法为纯神经网络模型（未使用统计先验特征），与多种序列推荐、时空推荐和 LLM 增强推荐方法进行了比较。

| 方法 | 来源 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SASRec | ICDM'18 | 40.38% | 48.28% | 28.98% | 31.26% | 27.01% |
| BERT4Rec | CIKM'19 | 39.60% | 46.70% | 29.03% | 30.94% | 27.11% |
| DuoRec | RecSys'22 | 34.72% | 40.69% | 24.54% | 26.34% | 22.76% |
| TiCoSeRec | SIGIR'23 | 40.69% | 48.08% | 29.65% | 32.04% | 27.54% |
| CrossDR | AAAI'22 | 40.80% | 47.20% | 30.67% | 32.59% | 28.66% |
| MAERec | WWW'22 | 37.65% | 44.53% | 27.72% | 29.85% | 26.10% |
| GETNext | KDD'22 | 43.60% | 51.30% | 32.56% | 35.13% | 30.97% |
| POIGDE | AAAI'23 | 34.11% | 38.36% | 26.64% | 27.97% | 25.40% |
| MTNet | AAAI'24 | 47.57% | 53.24% | 35.85% | 37.69% | 33.40% |
| DiffuRec | WWW'23 | 37.25% | 40.79% | 29.16% | 30.23% | 27.74% |
| LLM4POI | AAAI'24 | 44.64% | 46.66% | 37.49% | 38.16% | 35.90% |
| CoMaPOI | AAAI'25 | **51.62%** | **59.01%** | **40.42%** | **42.82%** | **37.67%** |
| **本文方法**（纯神经, 3轮） | — | 30.59% | 38.40% | 21.36% | 23.90% | 20.18% |
| **本文方法 + 统计先验**（3轮训练） | — | **38.57%** | **47.12%** | **27.91%** | **30.70%** | **26.37%** |
| **本文方法 + 统计先验**（1轮训练） | — | 37.74% | 46.36% | 27.62% | 30.43% | 26.21% |

> 说明：粗体为当前最优结果。本文纯神经结果反映多模态表示学习框架的自身能力（不使用统计先验特征）。"+ 统计先验"行展示在评估期叠加 9 种统计先验特征后（`configs/nyc_optim_priors.yaml`）的性能提升：NDCG@10 从 23.90% 提升至 30.70%（+6.80pp），验证了统计先验与神经表示的强互补性。

### 3.8 消融结论摘要

论文中的消融结果表明：

- 删除时间模态后 NDCG@10 下降最明显，说明用户签到节律是关键判别信号；
- 删除 SemAlign 后 NDCG@10 从 24.28% 降至 22.98%，证明拓扑—语义对齐有效；
- ContextFusion 优于简单拼接和固定权重融合，说明融合权重需要依赖轨迹上下文动态调整；
- 全量微调 GPT-2 略优于选择性微调，但收益很小；选择性微调以更低成本取得接近性能。

### 3.9 与最新方法的关系

当前方法的重点是构建可解释、可扩展、无泄漏的多模态表示学习底座，而不是通过候选收缩或复杂协同建模追求最高绝对指标。论文对比显示，CoMaPOI、MTNet、GETNext、LLM4POI 等方法在 NYC 上仍具有更高性能，主要原因包括：

- 本方法采用全空间候选匹配，尚未引入动态候选约束；
- 用户协同偏好和重复访问偏好建模仍较轻量；
- 语义模态当前主要是 POI 级文本，尚未扩展到用户画像或轨迹级语言化表达。

因此，后续可在当前框架上继续接入候选生成、候选重排序、用户协同建模、LoRA/Adapter 等参数高效微调策略。

## 4. 参考文件

- 实验协议：`protocol.md`
- 论文源码：`latex/paper.tex`
- 主实验配置：`configs/nyc_mvp.yaml`
- Debug 配置：`configs/debug.yaml`
- 模型实现：`poi_rec/models/`
- 训练与评估入口：`scripts/train.py`、`scripts/evaluate.py`

## 5. 性能优化路线图

当前模型在 NYC 测试集上的纯神经结果（NDCG@10=24.28%）与当前 SOTA 方法 CoMaPOI（NDCG@10=42.82%）存在约 18 个百分点的差距。以下从消融实验结论和对 SOTA 方法的差异分析出发，提出分层优化建议。

### P0 — 高优先级（预计 8-15 个百分点收益）

#### 5.1 统计先验端到端学习

当前统计先验仅在评估期作为手工加权分数叠加，权重通过手动人工调整。这种做法不仅难以达到最优配置，更严重地限制了神经网络从统计特征中学习高阶交互模式的能力。

**建议方案：**
1. 将 9 种统计先验的评分融合为一个**先验编码器**（PriorEncoder），对每个候选 POI 输出一个先验分数
2. 在训练期间启用 `apply_priors_during_training=true`，让神经打分与先验分数联合优化
3. 探索先验分数的非线性变换（如 MLP 投影 + sigmoid），替代当前简单的加权求和

**预期收益：** 1 轮训练 + 手工先验已从 24.28% 提升至 30.43% NDCG@10（+6.15pp）。如果端到端训练 5 轮，可望在此基础上再提升 3-5 个百分点。

**涉及文件：** `poi_rec/models/`（新增先验编码器模块）、`configs/nyc_mvp.yaml`（开启训练期先验）

#### 5.2 动态候选约束

当前模型采用全空间匹配（34,172 个候选 POI），而 CoMaPOI 等 SOTA 方法通过动态候选收缩大幅降低了候选空间大小和排序噪声。

**建议方案：**
1. **时间窗口候选过滤**：根据预测时间的小时/星期，只召回该时间槽内访问概率较高的 top-K POI（K=500~1000）
2. **空间网格候选过滤**：根据上一步 POI 的空间位置，限制候选在空间邻近范围内（如 5~10km 半径）
3. **用户历史候选增强**：用户历史访问过的 POI 始终加入候选集
4. 候选收缩后保留 500~2000 个候选，再用当前模型的全空间评分头进行排序

**预期收益：** 候选空间从 34k 缩小到 1~2k 后，HR@K 指标可望提升 5-10 个百分点（参考 CoMaPOI 候选收缩策略的效果）。

**涉及文件：** `poi_rec/data/`（候选生成逻辑）、`poi_rec/models/poi_model.py`（候选过滤接口）

### P1 — 中优先级（预计 3-8 个百分点收益）

#### 5.3 时间模态深化

消融实验表明，删除时间模态导致 NDCG@10 下降 1.89 个百分点，是所有模态中影响最大的。当前仅使用小时/星期/间隔三组嵌入求和，表征能力有限。

**建议方案：**
1. **多尺度时间编码**：引入绝对时间位置编码 + 相对时间间隔编码，类似 Transformer positional encoding 的周期延拓
2. **时间槽注意力**：对一天划分为 48 个时间槽（30 分钟粒度），学习每个用户在每个时间槽的 POI 偏好分布（类似 MTNet 的 mobility tree 思路）
3. **星期-小时交叉嵌入**：将星期与小时联合编码为一个 7×24 的二维嵌入矩阵

**预期收益：** NDCG@10 可望提升 1-2 个百分点。

**涉及文件：** `poi_rec/models/encoders.py`（TimeEncoder 增强）

#### 5.4 用户协同 / 重复访问显式建模

当前用户建模仅靠用户 ID embedding 和类别画像投影，缺乏显式的用户共访模式建模。

**建议方案：**
1. **用户-POI 协同过滤层**：用矩阵分解或 LightGCN 结构学习用户-POI 偏好矩阵
2. **重复访问激励**：当前 `history_repeat_prior` 只在统计先验中存在，应将其作为神经网络的显式偏置组件（如重复访问偏置项 `repeat_bias`）
3. **用户上下文融合加强**：在 ContextFusion 的门控计算中，将用户 ID embedding 的投影维度从当前 `hidden_dim` 增加到 `2×hidden_dim`

**预期收益：** NDCG@10 可望提升 1-2 个百分点。

**涉及文件：** `poi_rec/models/poi_model.py`（用户建模增强）、`poi_rec/models/fusion.py`（门控输入扩展）

#### 5.5 语义模态增强

当前语义模态使用固定 sentence-transformer 文本嵌入 + 可学习类别嵌入，未更新预训练文本编码器。

**建议方案：**
1. **POI 文本微调**：解冻 sentence-transformer 的最后 1-2 层，在下游 POI 推荐任务上进行端到端微调
2. **轨迹语义摘要**：将每个 POI 的文本描述扩展到**轨迹级语义**，例如对用户历史序列中若干 POI 的描述拼接后编码为轨迹语义
3. **区域语义嵌入**：基于 POI 的行政区域或功能分区（如曼哈顿下城区、中城区）学习区域级语义表示

**预期收益：** NDCG@10 可望提升 0.5-1.5 个百分点。

**涉及文件：** `poi_rec/models/encoders.py`（SemanticEncoder 增强）、`poi_rec/data/preprocess.py`（区域划分逻辑）

### P2 — 低优先级 / 探索性方向（预计 1-4 个百分点收益）

#### 5.6 模型容量与训练策略扩展

**建议方案：**
1. **增加 GPT-2 规模**：从当前 `gpt2`（124M）升级到 `gpt2-medium`（355M），同时设置 `gpt_freeze_policy: pathllm_selective` 控制参数量
2. **全量微调加正则化**：消融实验显示全量微调仅提升 0.13pp NDCG@10，但可能因 2 层 GPT 容量太小，在 4-6 层 GPT-2 上全量微调收益可能更大
3. **训练周期扩展**：当前 5 轮最优，可尝试 10-20 轮配合早停和学习率衰减，观察更深收敛点
4. **学习率 warmup + cosine decay**：替代当前固定学习率，让模型前期快速收敛、后期精细调优

**预期收益：** NDCG@10 可望提升 0.5-1.5 个百分点。

#### 5.7 数据增强与难例挖掘

**建议方案：**
1. **时间扰动**：对训练样本的时间戳进行小幅随机偏移（±1 小时），扩充时间模态多样性（类似 TiCoSeRec 的增强策略）
2. **POI dropout**：随机 mask 掉历史序列中 10-20% 的 POI，迫使模型利用剩余 POI 进行预测（类似 Cloze 任务）
3. **难例挖掘**：在批次内构造难负样本（如相同空间区域但不同类别的 POI 作为负例），提升判别能力

**预期收益：** NDCG@10 可望提升 0.5-1 个百分点。

#### 5.8 跨城市迁移与零样本验证

当前仅在 NYC 单数据集上验证，泛化性尚未充分证明。

**建议方案：**
1. 在 TKY（东京）和 CA（加州）数据集上重复预处理和实验流程
2. 验证 NYC 训练模型在 TKY/CA 上的零样本迁移效果
3. 如果迁移效果不佳，探索跨城市对齐损失（如域对抗训练或对比域适应）

**预期收益：** 验证模型稳健性，为后续部署提供支撑。

### 优先级汇总

| 优先级 | 方向 | 预期 NDCG@10 提升 | 建议实施顺序 |
| --- | --- | ---: | --- |
| P0 | 统计先验端到端学习 | +6~10pp | 1 |
| P0 | 动态候选约束 | +5~10pp | 2 |
| P1 | 时间模态深化 | +1~2pp | 3 |
| P1 | 用户协同与重复访问建模 | +1~2pp | 4 |
| P1 | 语义模态增强 | +0.5~1.5pp | 5 |
| P2 | 模型容量扩展 | +0.5~1.5pp | 6 |
| P2 | 数据增强 | +0.5~1pp | 7 |
| P2 | 跨城市验证 | 稳健性验证 | 8 |

> 说明：各方向预期收益为独立估计，实际叠加后可能存在边际递减效应。建议按 P0 → P1 → P2 的顺序逐步实验，每个方向完成后用消融实验确认增量贡献。
