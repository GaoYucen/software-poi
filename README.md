# 面向下一兴趣点推荐的跨模态自适应融合与检索（CAFR）

本仓库是论文《面向下一兴趣点推荐的跨模态自适应融合与检索方法》的实验与论文工程代码，聚焦 **Next POI Recommendation** 任务：根据用户历史签到轨迹，结合兴趣点的**语义、拓扑、空间及时序上下文**，预测用户下一次最可能访问的兴趣点。

当前仓库中的最新论文版本已将方法统一为 **CAFR（Cross-modal Adaptive Fusion and Retrieval）** 框架。相较于早期 README 中描述的 Path-LLM / GPT-2 / 静态先验融合方案，当前版本更强调：

- **多模态表示学习与跨模态对齐**
- **用户画像与上下文感知的自适应融合**
- **代价感知的自适应级联多索引检索**
- **用户画像条件化的时空序列建模**
- **候选重排序与联合优化**

论文源码位于 `latex/paper.tex`，方法细化说明位于 `latex/参考材料/POI推荐完整方法思路与公式.md`。

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 方法框架](#2-方法框架)
- [3. 数据集与论文实验设置](#3-数据集与论文实验设置)
- [4. 当前论文实验结果](#4-当前论文实验结果)
- [5. 仓库结构](#5-仓库结构)
- [6. 环境安装](#6-环境安装)
- [7. 快速运行](#7-快速运行)
- [8. 关键配置与脚本](#8-关键配置与脚本)
- [9. 与论文内容对应关系](#9-与论文内容对应关系)

---

## 1. 项目概述

### 1.1 任务定义

给定用户历史签到轨迹

\[
\mathcal{H}_u^{(n)}=[x_1^u,x_2^u,\ldots,x_n^u],
\]

以及与城市轨迹相关的多模态信息：

- `语义信息`：POI 类别、文本描述等
- `拓扑信息`：用户历史转移构建的 POI 转移图
- `空间信息`：POI 经纬度与相对位置关系
- `行为上下文`：时间槽、轨迹状态、用户偏好画像

模型目标是学习评分函数，对候选兴趣点进行排序并输出 Top-K 推荐结果。

### 1.2 当前方法主线

根据最新论文内容，CAFR 不再是“语言模型骨干 + 手工统计先验”的简单组合，而是一个围绕 **表示—融合—检索—排序** 构建的完整框架：

1. 对 POI 的语义、拓扑、空间信息进行统一编码；
2. 通过跨模态交互与一致性约束增强不同模态间的信息传递与对齐；
3. 利用用户画像和上下文，动态决定不同模态对当前用户/候选的贡献；
4. 通过代价感知级联检索，优先访问低代价候选源，仅在必要时访问高代价索引；
5. 对动态候选池执行时空序列建模和神经重排序。

### 1.3 本仓库包含的内容

本仓库同时包含：

- `poi_rec/`：核心数据处理、模型、损失、训练与评估代码
- `configs/`：调试、论文实验、候选检索等配置文件
- `scripts/`：预处理、训练、评估、结果汇总脚本
- `latex/`：论文源码、图表与补充材料
- `processed/`：已处理好的数据样例
- `tests/`：基础测试用例

---

## 2. 方法框架

### 2.1 总体流程

```text
原始签到序列 / POI 元数据
        │
        ├── 语义模态：类别嵌入 + 文本特征
        ├── 拓扑模态：转移统计 + Node2Vec
        ├── 空间模态：归一化经纬度
        │
        ▼
多模态表示学习与跨模态交互
        │
        ▼
跨模态一致性约束（InfoNCE 对齐损失）
        │
        ▼
用户画像编码 + 上下文感知自适应融合
        │
        ▼
代价感知的自适应级联多索引检索
        │
        ▼
用户画像条件化的时空序列建模
        │
        ▼
候选重排序与 Top-K 下一兴趣点推荐
```

### 2.2 五个核心模块

#### （1）多模态表示学习与跨模态对齐

对每个兴趣点分别构建：

- **语义表示**：类别嵌入 + 文本特征投影
- **拓扑表示**：转移统计特征 + Node2Vec 表示
- **空间表示**：归一化经纬度经 MLP 映射

随后通过跨模态交互模块在三种表示之间进行信息传递，并通过 InfoNCE 风格的对齐损失约束同一 POI 在不同模态空间中的表示接近。

#### （2）用户画像编码与上下文感知自适应融合

用户画像由以下信息组成：

- 类别偏好分布
- 时间偏好分布
- 空间活动中心与离散程度
- 历史访问 POI 分布

在此基础上，模型生成 **用户—候选粒度** 的模态权重，对语义、拓扑、空间表示进行动态加权融合，而不是采用固定权重。

#### （3）代价感知的自适应级联多索引检索

候选生成分为两个阶段：

- **Stage 1：低代价检索源**
  - history
  - user
  - transition
  - category
  - temporal
  - popularity

- **Stage 2：高代价检索源**
  - spatial
  - history_transition
  - trajectory_neighbor

模型根据第一阶段候选数量和分数熵判断是否需要访问第二阶段索引，从而在 **候选覆盖率** 与 **在线检索效率** 之间取得平衡。

#### （4）用户画像条件化的时空序列建模

在得到历史轨迹上各个 POI 的融合表示后，模型继续加入：

- 历史签到相对末次位置的空间关系编码
- 历史签到时间槽编码
- 目标时间槽与时间间隔编码

同时将用户画像作为条件前缀输入因果序列编码器，以获得用户当前时刻的动态查询表示。

#### （5）候选重排序与联合优化

在动态候选池内部，模型对每个候选构建：

- 候选融合表示
- 神经匹配分数
- 候选上下文特征分数

最终损失由：

- 排序损失 `L_rank`
- 跨模态对齐损失 `L_align`

联合组成：

\[
\mathcal{L}=\mathcal{L}_{rank}+\lambda_{align}\mathcal{L}_{align}
\]

---

## 3. 数据集与论文实验设置

### 3.1 数据集来源

论文当前使用两个真实城市签到数据集：

- **NYC**：纽约市签到数据
- **TKY**：东京签到数据

二者均来源于 Foursquare 公开签到数据，可用于下一兴趣点推荐任务。

### 3.2 论文中采用的整理后统计

根据 `latex/paper.tex` 当前版本，论文中的数据集统计如下：

| 数据集 | 用户数量 | POI 数量 | 训练样本数量 | 测试样本数量 |
| --- | ---: | ---: | ---: | ---: |
| NYC | 1,072 | 5,118 | 114,321 | 1,071 |
| TKY | 2,284 | 7,861 | 376,448 | 2,284 |

> 说明：这组数字是论文写作口径下的整理统计；与 `processed/` 中某些历史预处理产物的统计口径可能不完全一致。若以论文撰写为准，请优先参考 `latex/paper.tex`。

### 3.3 论文实验实现设置

当前论文实验描述中使用的关键设置包括：

- 框架：PyTorch
- 优化器：AdamW
- 学习率：`3e-4`
- 权重衰减：`1e-2`
- batch size：`256`
- dropout：`0.1`
- 隐空间维度：`128`
- 序列编码器：2 层 Transformer Encoder，4 个注意力头
- 最大历史长度：`30`
- 候选池大小：`256`
- Node2Vec 维度：`64`
- 文本语义特征：TF-IDF + Truncated SVD，64 维
- 对齐损失权重：`0.01`

### 3.4 评价指标

论文中主要采用：

- `HR@10`
- `NDCG@10`
- `MRR`

同时，在候选池构建分析中额外关注：

- Recall@N
- 平均检索延迟
- 高代价候选源调用率
- QPS

---

## 4. 当前论文实验结果

### 4.1 与代表性方法整体比较

根据 `latex/paper.tex` 当前表格，模型在 NYC / TKY 上的核心结果如下：

| Method | NYC HR@10 | NYC NDCG@10 | NYC MRR | TKY HR@10 | TKY NDCG@10 | TKY MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | 0.4828 | 0.3126 | 0.2701 | 0.4325 | 0.2837 | 0.2470 |
| BERT4Rec | 0.4670 | 0.3094 | 0.2711 | 0.4347 | 0.2908 | 0.2539 |
| DuoRec | 0.4069 | 0.2634 | 0.2276 | 0.3119 | 0.2089 | 0.1867 |
| TiCoSeRec | 0.4808 | 0.3204 | 0.2754 | 0.3704 | 0.2421 | 0.2151 |
| CrossDR | 0.4720 | 0.3259 | 0.2866 | 0.4057 | 0.2619 | 0.2265 |
| MAERec | 0.4453 | 0.2985 | 0.2610 | 0.4066 | 0.2614 | 0.2305 |
| GETNext | 0.5130 | 0.3513 | 0.3097 | 0.4850 | 0.3321 | 0.2921 |
| POIGDE | 0.3836 | 0.2797 | 0.2540 | 0.3182 | 0.2357 | 0.2180 |
| DiffuRec | 0.4079 | 0.3023 | 0.2774 | 0.3989 | 0.2870 | 0.2594 |
| LLM4POI | 0.4666 | 0.3816 | **0.3590** | 0.3948 | 0.2822 | 0.2972 |
| **Ours (CAFR)** | **0.5257** | **0.3828** | 0.3442 | **0.5346** | **0.3574** | **0.3110** |

从论文当前结果看，CAFR 在两个真实城市数据集上都取得了较强的命中能力和排序质量，尤其在 `HR@10` 和 `NDCG@10` 上表现突出。

### 4.2 代价感知候选池构建的效率收益

论文还专门分析了 CAFR 的工程收益：

- 与固定访问全部高代价候选源的方法相比，CAFR 在候选召回仅轻微下降的前提下，显著降低平均检索延迟；
- 高代价候选源调用率由 `100%` 降至 `68.25%`；
- QPS 从 `287.96` 提升到 `341.55`；
- 最终推荐质量保持基本稳定。

这部分结果对应论文中的“代价感知候选池构建分析”章节与图 `latex/figure/nyc_cost_aware_candidate_pool.pdf`。

---

## 5. 仓库结构

```text
software-poi/
├── README.md
├── protocol.md
├── pyproject.toml
├── configs/
│   ├── debug.yaml
│   ├── paper_nyc.yaml
│   ├── paper_tky.yaml
│   └── ...
├── data/
├── processed/
├── poi_rec/
│   ├── data/
│   │   ├── candidates.py
│   │   ├── dataset.py
│   │   └── preprocess.py
│   ├── losses/
│   │   ├── alignment_loss.py
│   │   └── recommendation_loss.py
│   ├── models/
│   │   ├── alignment.py
│   │   ├── encoders.py
│   │   ├── fusion.py
│   │   ├── poi_model.py
│   │   └── priors.py
│   ├── training/
│   └── utils/
├── scripts/
│   ├── preprocess.py
│   ├── pretrain_alignment.py
│   ├── train.py
│   ├── evaluate.py
│   └── summarize_paper_results.py
├── tests/
└── latex/
    ├── paper.tex
    ├── references.bib
    ├── figure/
    └── 参考材料/
```

### 5.1 核心代码文件

| 文件 | 作用 |
| --- | --- |
| `poi_rec/models/encoders.py` | 语义、拓扑、空间、时间、画像等编码器 |
| `poi_rec/models/fusion.py` | 跨模态交互与自适应融合相关模块 |
| `poi_rec/models/alignment.py` | 跨模态对齐投影/对齐逻辑 |
| `poi_rec/models/poi_model.py` | 总模型与候选打分主逻辑 |
| `poi_rec/data/candidates.py` | 多源候选生成与动态候选池构建 |
| `poi_rec/losses/alignment_loss.py` | 对齐损失 |
| `poi_rec/losses/recommendation_loss.py` | 推荐排序损失 |
| `scripts/train.py` | 主训练入口 |
| `scripts/evaluate.py` | 评估入口 |

---

## 6. 环境安装

建议使用 Python 3.11。当前远程环境推荐直接使用 `py11` 解释器：

```bash
/opt/conda/envs/py11/bin/python -m pip install -e .
```

如果仅手动安装依赖，可使用：

```bash
/opt/conda/envs/py11/bin/python -m pip install torch transformers pandas numpy scikit-learn networkx node2vec gensim pyyaml tqdm safetensors tokenizers
```

若实验配置依赖 HuggingFace 模型，请确保运行环境能访问或提前缓存对应模型文件。

---

## 7. 快速运行

以下命令均默认使用：`/opt/conda/envs/py11/bin/python`

### 7.1 Debug / Smoke Test

```bash
/opt/conda/envs/py11/bin/python scripts/preprocess.py --config configs/debug.yaml
/opt/conda/envs/py11/bin/python scripts/pretrain_alignment.py --config configs/debug.yaml
/opt/conda/envs/py11/bin/python scripts/train.py --config configs/debug.yaml
/opt/conda/envs/py11/bin/python scripts/evaluate.py --checkpoint runs/debug/best.pt --split val
```

### 7.2 论文配置示例

如果使用论文主配置，可参考：

```bash
/opt/conda/envs/py11/bin/python scripts/preprocess.py --config configs/paper_nyc.yaml
/opt/conda/envs/py11/bin/python scripts/pretrain_alignment.py --config configs/paper_nyc.yaml
/opt/conda/envs/py11/bin/python scripts/train.py --config configs/paper_nyc.yaml
/opt/conda/envs/py11/bin/python scripts/evaluate.py --checkpoint runs/paper_nyc/best.pt --split test
```

> 注意：不同阶段曾存在多套实验配置（如 `nyc_mvp.yaml`、`paper_nyc.yaml`、`paper_tky.yaml` 等）。若目标是复现当前论文文本，请优先以 `latex/paper.tex` 中描述的实验设置为准，并结合实际本地配置文件核对。

### 7.3 运行测试

```bash
/opt/conda/envs/py11/bin/python -m unittest discover -s tests
```

---

## 8. 关键配置与脚本

### 8.1 常见配置文件

| 配置文件 | 用途 |
| --- | --- |
| `configs/debug.yaml` | 小规模调试 |
| `configs/paper_nyc.yaml` | 论文 NYC 配置 |
| `configs/paper_tky.yaml` | 论文 TKY 配置 |
| `configs/nyc_dynamic_candidates.yaml` | 动态候选相关配置 |
| `configs/nyc_p0_optim.yaml` | 先验/候选相关实验配置 |

### 8.2 常见脚本

| 脚本 | 功能 |
| --- | --- |
| `scripts/preprocess.py` | 数据预处理 |
| `scripts/pretrain_alignment.py` | 对齐预训练 |
| `scripts/train.py` | 主训练 |
| `scripts/evaluate.py` | 模型评估 |
| `scripts/generate_experiment_figures.py` | 生成实验图 |
| `scripts/summarize_paper_results.py` | 汇总论文结果 |

---

## 9. 与论文内容对应关系

如果你是从论文或 LaTeX 材料进入本仓库，建议按下面顺序阅读：

1. **论文主文**：`latex/paper.tex`
2. **方法公式与撰写说明**：`latex/参考材料/POI推荐完整方法思路与公式.md`
3. **实验图表**：`latex/figure/`
4. **核心实现**：`poi_rec/models/`、`poi_rec/data/`
5. **训练评估入口**：`scripts/train.py`、`scripts/evaluate.py`

### 9.1 当前 README 的定位

本 README 已根据最新论文内容更新，重点反映：

- 方法名称已统一为 **CAFR**；
- 核心创新点已更新为 **跨模态交互 + 自适应融合 + 代价感知级联检索**；
- 实验结果以 `latex/paper.tex` 当前版本为准；
- 旧版 README 中对 Path-LLM、GPT-2 静态骨干、手工统计先验主导方案的描述不再作为本文档主线。

### 9.2 补充说明

由于仓库中保留了多个阶段的实验配置和历史实现，代码层面仍可能看到一些旧模块名、探索性配置或未完全清理的实验脚本。这是正常现象。若你的目标是：

- **复现当前论文写作内容**：优先看 `latex/paper.tex` 与 `latex/参考材料/POI推荐完整方法思路与公式.md`
- **理解现有工程代码结构**：优先看 `poi_rec/` 与 `scripts/`
- **继续扩展实验**：优先看 `configs/`、`scripts/` 与 `latex/results/`

---

如需进一步同步 README 与某个特定配置文件、训练脚本或最终投稿版本，我可以继续帮你做第二轮精修，例如补充“配置—论文章节—代码模块”的逐项映射表。
