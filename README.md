# 基于多模态数据融合的下一兴趣点推荐

本仓库提供论文《基于多模态数据融合的下一兴趣点推荐方法》的核心实验代码。任务目标是根据用户历史签到序列预测下一次访问的 POI（Point-of-Interest）。当前实现面向 TSMC2014/Foursquare 签到数据，主实验使用 NYC 数据集，并采用时间顺序切分、训练集构图和闭世界全排序评估协议，避免行为统计信息泄漏。

代码实现了一个 Path-LLM 风格的多模态 POI 表示学习框架：从签到日志和 POI 元数据构建拓扑、语义、空间、时间等模态，通过 SemAlign 做拓扑—语义对齐，通过 ContextFusion 做上下文感知融合，最后使用选择性微调的 GPT-2 骨干网络完成序列建模和下一 POI 匹配。

## 目录

- [1. 仓库内容与 GitHub 上传说明](#1-仓库内容与-github-上传说明)
- [2. 模型结构](#2-模型结构)
- [3. 数据集与实验协议](#3-数据集与实验协议)
- [4. 环境安装](#4-环境安装)
- [5. 快速运行](#5-快速运行)
- [6. 配置文件说明](#6-配置文件说明)
- [7. 当前性能结果](#7-当前性能结果)
- [8. 进阶优化模块](#8-进阶优化模块)
- [9. 参考文件](#9-参考文件)

## 1. 仓库内容与 GitHub 上传说明

### 1.1 程序运行必要文件

建议上传并保留以下文件/目录，clone 后即可复现预处理、训练、评估和测试流程：

| 路径 | 作用 | 是否必要 |
| --- | --- | --- |
| `poi_rec/` | 数据处理、模型、损失、训练和评估核心代码 | 必要 |
| `scripts/` | 命令行入口：预处理、预训练、训练、评估、Reranker 训练 | 必要 |
| `configs/` | Debug、NYC 主实验、统计先验、动态候选等配置 | 必要 |
| `tests/` | 单元测试 | 建议保留 |
| `pyproject.toml` | Python 包和依赖声明 | 必要 |
| `README.md` | 项目说明和运行命令 | 必要 |
| `protocol.md` | 数据切分和无泄漏实验协议 | 必要 |
| `data/dataset_TSMC2014_NYC.txt` | NYC 原始签到数据 | 复现实验必要 |
| `data/dataset_TSMC2014_TKY.txt` | TKY 原始签到数据 | 跨城市实验可选 |
| `processed/NYC/` | 已处理 NYC 数据，可跳过预处理直接训练/评估 | 建议保留 |
| `runs/nyc_mvp/` | 主实验 baseline checkpoint，可直接评估 | 建议保留 |
| `latex/paper.tex`、`latex/paper.pdf`、`latex/figure/` | 论文源码、PDF 和图片 | 论文归档需要时保留 |

### 1.2 不建议上传的本地产物

以下文件通常不是程序运行必要文件，建议忽略或不提交：

- Python 缓存：`__pycache__/`、`*.pyc`
- 本地调试数据：`processed/debug*/`
- 本地训练日志和临时实验：`runs/debug*/`、`runs/*_local/`、`runs/nyc_4090*/`、`runs/*benchmark*/`
- 优化实验产生的大 checkpoint 和日志：`runs/nyc_optim*/`、`runs/nyc_p0_optim/`、`runs/nyc_small_candidate_reranker*/`
- 训练状态文件：`*.pid`
- 日志文件：`*.log`、`*_output.log`、`nohup*.out`
- LaTeX 编译中间文件：`latex/*.aux`、`latex/*.bbl`、`latex/*.bcf`、`latex/*.blg`、`latex/*.fdb_latexmk`、`latex/*.fls`、`latex/*.log`、`latex/*.run.xml`、`latex/*.xdv`

### 1.3 Git LFS

仓库中包含原始数据、处理后数据和 baseline checkpoint 等大文件，建议使用 Git LFS 管理。首次 clone 后执行：

```bash
git lfs install
git lfs pull
```

当前 `.gitattributes` 已将以下关键大文件纳入 LFS：

- `data/dataset_TSMC2014_NYC.txt`
- `data/dataset_TSMC2014_TKY.txt`
- `processed/NYC/arrays.npz`
- `processed/NYC/train.json`
- `processed/NYC/val.json`
- `processed/NYC/test.json`
- `runs/nyc_mvp/best.pt`
- `runs/nyc_mvp/alignment_pretrain.pt`

如需额外上传新的 checkpoint，例如 `runs/nyc_p0_optim/best.pt`，应先加入 Git LFS，再提交。

## 2. 模型结构

整体流程如下：

```text
原始签到序列 / POI 元数据
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

核心实现文件：

- `poi_rec/models/poi_model.py`：总模型、候选打分、统计先验融合
- `poi_rec/models/encoders.py`：拓扑、语义、空间、时间和用户画像编码器
- `poi_rec/models/alignment.py`：SemAlign 投影头
- `poi_rec/models/fusion.py`：ContextFusion / DynamicFusion
- `poi_rec/models/gpt_backbone.py`：GPT-2 序列骨干
- `poi_rec/models/priors.py`：PriorEncoder 和 CandidateReranker
- `poi_rec/data/candidates.py`：动态候选生成器

### 2.1 多模态编码器

| 模态 | 输入来源 | 实现方式 |
| --- | --- | --- |
| 拓扑模态 | 训练集相邻签到诱导的有向加权 POI 转移图 | 转移统计特征 + Node2Vec 嵌入，经线性投影和 LayerNorm 得到拓扑表示 |
| 语义模态 | POI 类别、城市和文本描述 | `sentence-transformers/all-MiniLM-L6-v2` 生成文本嵌入，叠加可学习类别嵌入 |
| 空间模态 | POI 经纬度 | 坐标归一化后经 MLP/GELU/LayerNorm 投影到隐藏空间 |
| 时间模态 | 签到时间上下文 | 本地小时、星期、时间间隔 bucket 分别嵌入后求和并归一化 |
| 用户/画像信息 | 训练集用户类别偏好统计、用户 ID | 用户 ID embedding 与用户类别画像投影作为辅助上下文 |

所有行为统计类特征均只由训练集构建，包括转移图、Node2Vec、流行度、用户-POI 偏好和用户-类别偏好。

### 2.2 SemAlign 与 ContextFusion

SemAlign 通过 `topology_projection` 和 `semantic_projection` 两个投影头对齐拓扑表示与语义表示，训练时结合实例级和特征级 InfoNCE 损失。

ContextFusion 根据当前序列上下文动态生成门控向量，而不是简单拼接或固定加权：

```text
gate = sigmoid(MLP([topology, semantic, temporal, user, profile]))
static = gate * topology + (1 - gate) * semantic
token = MLP([static, spatial, temporal, user, profile])
```

### 2.3 GPT-2 序列建模与候选匹配

融合后的 token 序列输入 GPT-2 骨干网络。主实验配置使用：

- `gpt_model_name: gpt2`
- `hidden_dim: 128`
- `gpt_layers: 2`
- `gpt_heads: 4`
- `max_seq_len: 20`
- `gpt_freeze_policy: pathllm_selective`

推理时取最后一个有效位置的隐藏状态作为查询向量，与候选 POI 表示做归一化点积匹配。主实验使用 `closed_world` 协议，即候选和评估目标限制在训练集中出现过的 POI。

## 3. 数据集与实验协议

### 3.1 数据来源

数据来自 TSMC2014/Foursquare 签到数据，原始字段包括用户 ID、POI/venue ID、类别 ID 与类别名称、纬度、经度、时区偏移和 UTC 签到时间。

当前仓库主要包含 NYC 数据配置：

- 原始文件：`data/dataset_TSMC2014_NYC.txt`
- 处理后目录：`processed/NYC`
- 主实验配置：`configs/nyc_mvp.yaml`

### 3.2 预处理与切分协议

预处理遵循 `protocol.md`：

1. 按用户对签到记录进行 UTC 时间排序；
2. 使用时区偏移计算本地小时和星期；
3. 过滤签到次数少于 2 的用户；
4. 按用户时间顺序切分训练/验证/测试集：80% / 10% / 10%；
5. 每个目标位置构造一个下一 POI 样本，历史长度最多为 `max_seq_len=20`；
6. 所有拓扑、流行度和用户偏好统计仅由训练前缀生成；
7. 主实验采用闭世界全排序，只评估训练集中出现过的 POI 目标。

### 3.3 当前处理后 NYC 数据统计

根据 `processed/NYC/metadata.json`：

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

论文正文中的数据统计表使用了另一个整理口径。当前 README 以代码目录中最新 `processed/NYC/metadata.json` 为准。

## 4. 环境安装

建议使用 Python 3.11。当前远程服务器可直接使用 `py11` 环境：

```bash
/opt/conda/envs/py11/bin/python -m pip install -e .
```

如果不使用 editable install，也可以安装主要依赖：

```bash
/opt/conda/envs/py11/bin/python -m pip install torch transformers pandas numpy scikit-learn networkx node2vec gensim pyyaml tqdm safetensors tokenizers
```

正式 NYC 配置需要能够加载以下 HuggingFace 模型：

- `gpt2`
- `sentence-transformers/all-MiniLM-L6-v2`

如果运行环境无法联网下载模型，需要提前缓存到本地。`fallback_to_random_gpt: true` 只建议用于 debug 或 smoke test，不建议用于正式实验。

## 5. 快速运行

以下命令均使用项目约定的 Python 解释器：`/opt/conda/envs/py11/bin/python`。

### 5.1 Debug Smoke Test

```bash
/opt/conda/envs/py11/bin/python scripts/preprocess.py --config configs/debug.yaml
/opt/conda/envs/py11/bin/python scripts/pretrain_alignment.py --config configs/debug.yaml
/opt/conda/envs/py11/bin/python scripts/train.py --config configs/debug.yaml
/opt/conda/envs/py11/bin/python scripts/evaluate.py --checkpoint runs/debug/best.pt --split val
```

### 5.2 NYC 主实验

如果 `processed/NYC/` 已存在，可跳过第一步预处理。

```bash
/opt/conda/envs/py11/bin/python scripts/preprocess.py --config configs/nyc_mvp.yaml
/opt/conda/envs/py11/bin/python scripts/pretrain_alignment.py --config configs/nyc_mvp.yaml
/opt/conda/envs/py11/bin/python scripts/train.py --config configs/nyc_mvp.yaml
/opt/conda/envs/py11/bin/python scripts/evaluate.py --checkpoint runs/nyc_mvp/best.pt --split test
```

### 5.3 直接评估已上传 baseline checkpoint

如果已通过 Git LFS 拉取 `runs/nyc_mvp/best.pt`：

```bash
/opt/conda/envs/py11/bin/python scripts/evaluate.py --checkpoint runs/nyc_mvp/best.pt --split test
```

### 5.4 统计先验 / P0 优化实验

手工统计先验配置：

```bash
/opt/conda/envs/py11/bin/python scripts/pretrain_alignment.py --config configs/nyc_4090.yaml
/opt/conda/envs/py11/bin/python scripts/train.py --config configs/nyc_4090.yaml
/opt/conda/envs/py11/bin/python scripts/evaluate.py --checkpoint runs/nyc_4090_epoch1/best.pt --split test
```

PriorEncoder 端到端先验配置：

```bash
/opt/conda/envs/py11/bin/python scripts/pretrain_alignment.py --config configs/nyc_p0_optim.yaml
/opt/conda/envs/py11/bin/python scripts/train.py --config configs/nyc_p0_optim.yaml
/opt/conda/envs/py11/bin/python scripts/evaluate.py --checkpoint runs/nyc_p0_optim/best.pt --split test
```

动态候选约束配置：

```bash
/opt/conda/envs/py11/bin/python scripts/evaluate.py --checkpoint runs/nyc_p0_optim/best.pt --split test --config configs/nyc_dynamic_candidates.yaml
```

### 5.5 单元测试

```bash
/opt/conda/envs/py11/bin/python -m unittest discover -s tests
```

## 6. 配置文件说明

| 配置 | 作用 |
| --- | --- |
| `configs/debug.yaml` | 小规模 debug / smoke test |
| `configs/nyc_mvp.yaml` | NYC 主实验配置，纯神经多模态模型 |
| `configs/nyc_4090.yaml` | RTX 4090 快速配置，评估期叠加手工统计先验 |
| `configs/nyc_optim_priors.yaml` | 统计先验权重优化实验配置 |
| `configs/nyc_p0_optim.yaml` | PriorEncoder 端到端先验学习配置 |
| `configs/nyc_dynamic_candidates.yaml` | 动态候选约束评估配置 |
| `configs/nyc_small_candidate_reranker.yaml` | 小候选集 CandidateReranker 实验配置 |
| `configs/nyc_profile_llm.yaml` | 用户画像 / LLM 相关探索配置 |

主配置 `configs/nyc_mvp.yaml` 的关键参数：

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

## 7. 当前性能结果

### 7.1 主实验结果

论文中报告的完整模型在 NYC 测试集上的主结果如下：

| 指标 | 验证集 | 测试集 |
| --- | ---: | ---: |
| HR@5 | 31.14% | 30.70% |
| NDCG@5 | 21.83% | 22.01% |
| HR@10 | 39.05% | 37.66% |
| NDCG@10 | 24.41% | 24.28% |
| HR@20 | 45.36% | 43.34% |
| MRR | 20.60% | 20.80% |

训练过程中验证集最佳 epoch 为第 4 轮。验证集与测试集 NDCG@10 差距约 0.13 个百分点，说明当前无泄漏切分下泛化较稳定。

### 7.2 与统计先验结合后的结果

| 方法 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本文方法（纯神经，3 轮） | 30.59% | 38.40% | 21.36% | 23.90% | 20.18% |
| 本文方法 + 统计先验（3 轮训练） | **38.57%** | **47.12%** | **27.91%** | **30.70%** | **26.37%** |
| 本文方法 + 统计先验（1 轮训练） | 37.74% | 46.36% | 27.62% | 30.43% | 26.21% |
| PriorEncoder（1 轮训练） | 37.72% | 47.07% | 27.11% | 30.15% | 25.68% |
| PriorEncoder + 动态候选约束 | 37.75% | 47.18% | 27.12% | 30.19% | 25.84% |

统计先验显著提升召回和排序指标，说明 POI 推荐任务中流行度、转移、重复访问、用户偏好和空间邻近等统计信号与神经表示具有强互补性。

## 8. 进阶优化模块

### 8.1 Statistical Priors

模型支持在神经匹配分数之外叠加统计先验：

| 先验 | 配置参数 |
| --- | --- |
| POI 全局流行度 | `popularity_prior_weight` |
| 最近转移 | `transition_prior_weight` |
| 历史多步转移 | `history_transition_prior_weight` |
| 用户-POI 偏好 | `user_poi_prior_weight` |
| 用户类别偏好 | `user_category_prior_weight` |
| 类别转移 | `category_transition_prior_weight` |
| 重复访问 | `history_repeat_prior_weight` |
| 空间邻近 | `spatial_prior_weight` |
| 共访偏好 | `co_visit_prior_weight` |

最终分数：

```text
final_score = neural_score + Σ(weight_i * statistical_prior_i)
```

### 8.2 PriorEncoder

`poi_rec/models/priors.py` 中的 `PriorEncoder` 可学习组合密集先验分数。在 `configs/nyc_p0_optim.yaml` 中启用：

```yaml
use_prior_encoder: true
apply_priors_during_training: true
```

为避免大候选集下显存溢出，当前实现使用轻量单层 `Linear(4, 1, bias=False)` 组合 4 个密集先验：popularity、category_transition、user_category、spatial。其余稀疏先验仍采用原有加权逻辑。

### 8.3 动态候选约束

`poi_rec/data/candidates.py` 中的 `DynamicCandidateGenerator` 可在评估时将候选空间从训练可见 POI 收缩到固定规模候选集，例如 2,000 个候选。候选来源包括：

- 最近 POI 转移 top-K
- 历史多步转移 top-K
- 用户训练历史 POI
- 上一步 POI 的空间近邻
- 小时/星期时间槽热门 POI
- 全局热门 POI 补足

配置见 `configs/nyc_dynamic_candidates.yaml`。

### 8.4 CandidateReranker

当前已实现小候选集精排的工程基础：

- `poi_rec/models/priors.py`：`CandidateReranker`，10 维特征 MLP
- `poi_rec/models/poi_model.py`：`scores_for_candidates()` 候选级接口
- `scripts/train_reranker.py`：冻结 backbone 的独立 Reranker 训练脚本

后续优化重点可从全空间神经匹配转向：高质量小候选召回 + 候选集内可学习精排 + 相似轨迹/用户意图/统计先验联合建模。

## 9. 参考文件

- 实验协议：`protocol.md`
- 主实验配置：`configs/nyc_mvp.yaml`
- Debug 配置：`configs/debug.yaml`
- P0 优化配置：`configs/nyc_p0_optim.yaml`
- 动态候选配置：`configs/nyc_dynamic_candidates.yaml`
- 模型实现：`poi_rec/models/`
- 数据处理：`poi_rec/data/`
- 训练与评估入口：`scripts/train.py`、`scripts/evaluate.py`
- 论文源码：`latex/paper.tex`
