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

### 1.4 GPT-2 序列建模与候选匹配

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

### 3.4 测试

```bash
conda run -n py11 python -m unittest discover -s tests
```

### 3.5 当前性能结果

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

### 3.6 消融结论摘要

论文中的消融结果表明：

- 删除时间模态后 NDCG@10 下降最明显，说明用户签到节律是关键判别信号；
- 删除 SemAlign 后 NDCG@10 从 24.28% 降至 22.98%，证明拓扑—语义对齐有效；
- ContextFusion 优于简单拼接和固定权重融合，说明融合权重需要依赖轨迹上下文动态调整；
- 全量微调 GPT-2 略优于选择性微调，但收益很小；选择性微调以更低成本取得接近性能。

### 3.7 与最新方法的关系

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
