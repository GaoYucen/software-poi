# 面向下一兴趣点推荐的多模态表示、画像条件化序列建模与代价感知自适应检索方法

> 本文档面向论文“方法”章节撰写，按照当前模型图与代码实现梳理完整技术路线。  
> 其中，多模态编码、用户画像、自适应融合、时空序列建模、候选重排序与当前公开代码一致；“代价感知的自适应级联多索引检索”按照当前已确定的设计方案描述。若论文要求与公开仓库逐行一致，需在提交前再次核对本地尚未推送的级联检索实现。

---

## 1. 问题定义

设用户集合、兴趣点集合和签到记录集合分别为

\[
\mathcal U=\{u_1,u_2,\ldots,u_{|\mathcal U|}\},
\qquad
\mathcal P=\{p_1,p_2,\ldots,p_{|\mathcal P|}\},
\]

\[
\mathcal R=\{(u,p,t,\boldsymbol l_p,c_p,\boldsymbol x_p)\},
\]

其中：

- \(u\in\mathcal U\) 表示用户；
- \(p\in\mathcal P\) 表示兴趣点；
- \(t\) 表示签到时间；
- \(\boldsymbol l_p=(\mathrm{lat}_p,\mathrm{lon}_p)\) 表示兴趣点坐标；
- \(c_p\) 表示兴趣点类别；
- \(\boldsymbol x_p\) 表示名称、类别、文本描述等语义属性。

对于用户 \(u\)，按时间顺序排列其历史签到序列：

\[
\mathcal S_u^{t-1}
=
\big[
(p_1,t_1),
(p_2,t_2),
\ldots,
(p_{t-1},t_{t-1})
\big].
\]

给定历史签到序列 \(\mathcal S_u^{t-1}\)、用户画像及目标签到时间条件，下一兴趣点推荐任务旨在估计任意候选兴趣点 \(p\) 成为下一签到位置的得分：

\[
s(u,p\mid \mathcal S_u^{t-1},t_t,\Delta t_t),
\]

并返回得分最高的 \(K\) 个兴趣点：

\[
\hat{\mathcal P}_{u,t}^{K}
=
\operatorname{TopK}_{p\in\mathcal P}
s(u,p\mid \mathcal S_u^{t-1},t_t,\Delta t_t).
\]

直接在完整兴趣点集合上计算所有候选得分的成本较高。本文首先通过代价感知的自适应级联多索引检索构建动态候选池

\[
\mathcal C_u\subseteq\mathcal P,\qquad |\mathcal C_u|\ll |\mathcal P|,
\]

再在 \(\mathcal C_u\) 中完成神经匹配和候选重排序：

\[
\hat{\mathcal P}_{u,t}^{K}
=
\operatorname{TopK}_{p\in\mathcal C_u}
s(u,p).
\]

---

## 2. 方法总体框架

本文方法由五个核心部分组成：

1. **多模态表示学习与跨模态对齐**：分别编码兴趣点语义、拓扑和空间信息，并通过跨模态交互和对齐损失获得互补表征；
2. **上下文感知的自适应融合**：以用户画像为条件，动态估计不同模态对当前用户的重要性，生成融合兴趣点表示；
3. **代价感知的自适应级联多索引检索**：先执行低代价多源召回，再根据候选数量和分数熵决定是否启用高代价检索源；
4. **用户画像条件化的时空序列建模**：将用户画像作为序列前缀，对时空增强的历史签到序列进行因果建模，获得用户动态查询表示；
5. **候选重排序与下一兴趣点推荐**：联合神经匹配得分、检索得分、来源特征及候选上下文特征，对动态候选池进行重排序。

整体计算过程可写为：

\[
\{\boldsymbol e_p^{sem},
\boldsymbol e_p^{topo},
\boldsymbol e_p^{spa}\}
\xrightarrow{\text{跨模态交互}}
\{
\widetilde{\boldsymbol e}_p^{sem},
\widetilde{\boldsymbol e}_p^{topo},
\widetilde{\boldsymbol e}_p^{spa}
\},
\]

\[
\boldsymbol l_u
\xrightarrow{\text{画像条件门控}}
\boldsymbol h_p,
\]

\[
\mathcal S_u^{t-1}
\xrightarrow{\text{自适应多索引检索}}
\mathcal C_u,
\]

\[
(\mathcal S_u^{t-1},\boldsymbol l_u,t_t,\Delta t_t)
\xrightarrow{\text{因果序列建模}}
\boldsymbol q_u,
\]

\[
(\mathcal C_u,\boldsymbol q_u,\boldsymbol h_p)
\xrightarrow{\text{候选重排序}}
\hat{\mathcal P}_{u,t}^{K}.
\]

---

# 3. 多模态表示学习与跨模态对齐

## 3.1 语义模态编码

兴趣点语义信息包括类别、名称及预计算文本表示。设兴趣点 \(p\) 的类别编号为 \(c_p\)，文本特征为 \(\boldsymbol x_p^{text}\)。类别嵌入与文本特征分别投影至统一隐空间：

\[
\boldsymbol e_{p,c}
=
\boldsymbol E_c[c_p],
\]

\[
\boldsymbol e_{p,x}
=
\operatorname{MLP}_{text}
(\boldsymbol x_p^{text}).
\]

语义表示为：

\[
\boldsymbol e_p^{sem}
=
\operatorname{LN}
\left(
\operatorname{MLP}_{sem}
\left[
\boldsymbol e_{p,c};
\boldsymbol e_{p,x}
\right]
\right),
\]

其中 \([\cdot;\cdot]\) 表示向量拼接，\(\operatorname{LN}\) 表示层归一化。

若实现中直接将类别嵌入和文本投影相加，则可改写为：

\[
\boldsymbol e_p^{sem}
=
\operatorname{LN}
\left(
\boldsymbol e_{p,c}
+
\boldsymbol e_{p,x}
\right).
\]

---

## 3.2 拓扑模态编码

从训练集签到数据构建有向兴趣点转移图：

\[
\mathcal G=(\mathcal P,\mathcal E),
\]

其中，当训练集中出现从 \(p_i\) 到 \(p_j\) 的连续转移时，建立边 \((p_i,p_j)\)。边权可由转移频次或归一化转移概率给出：

\[
w_{ij}
=
\#(p_i\rightarrow p_j),
\]

\[
P(p_j\mid p_i)
=
\frac{w_{ij}}
{\sum_{p_k}w_{ik}}.
\]

对每个兴趣点，代码同时使用转移统计特征和 Node2Vec 表示。设二者分别为

\[
\boldsymbol x_p^{trans},
\qquad
\boldsymbol x_p^{n2v},
\]

则拓扑编码为：

\[
\boldsymbol e_p^{topo}
=
\operatorname{LN}
\left(
\operatorname{MLP}_{topo}
\left[
\boldsymbol x_p^{trans};
\boldsymbol x_p^{n2v}
\right]
\right).
\]

拓扑表示刻画兴趣点在全局移动网络中的结构角色及邻接转移模式。

---

## 3.3 空间模态编码

设兴趣点 \(p\) 的归一化地理坐标为

\[
\boldsymbol l_p
=
[\mathrm{lat}_p,\mathrm{lon}_p].
\]

通过多层感知机获得空间表示：

\[
\boldsymbol e_p^{spa}
=
\operatorname{LN}
\left(
\operatorname{MLP}_{spa}
(\boldsymbol l_p)
\right).
\]

该表示编码兴趣点的绝对空间位置。历史签到序列中的相对位置关系将在后续时空序列建模阶段进一步刻画。

---

## 3.4 跨模态注意力交互

语义、拓扑和空间模态分别描述“兴趣点是什么”“兴趣点如何与其他位置连接”以及“兴趣点位于何处”。为使不同模态能够相互补充，本文采用跨模态交互模块：

\[
\left(
\widetilde{\boldsymbol e}_p^{sem},
\widetilde{\boldsymbol e}_p^{topo},
\widetilde{\boldsymbol e}_p^{spa}
\right)
=
\operatorname{CMI}
\left(
\boldsymbol e_p^{sem},
\boldsymbol e_p^{topo},
\boldsymbol e_p^{spa}
\right),
\]

其中 \(\operatorname{CMI}\) 表示跨模态交互函数。

一种可用于论文描述的多头注意力形式为：

\[
\operatorname{Attn}
(\boldsymbol Q,\boldsymbol K,\boldsymbol V)
=
\operatorname{softmax}
\left(
\frac{\boldsymbol Q\boldsymbol K^\top}
{\sqrt d}
\right)
\boldsymbol V.
\]

以语义模态为例，其交互表示可写为：

\[
\widetilde{\boldsymbol e}_p^{sem}
=
\operatorname{LN}
\left(
\boldsymbol e_p^{sem}
+
\operatorname{Attn}
\left(
\boldsymbol e_p^{sem},
[\boldsymbol e_p^{topo};\boldsymbol e_p^{spa}],
[\boldsymbol e_p^{topo};\boldsymbol e_p^{spa}]
\right)
\right).
\]

拓扑模态和空间模态采用相同形式。实际代码中的具体残差、前馈网络和多头配置应以 `poi_rec/models/fusion.py` 为准。

---

## 3.5 跨模态对齐损失

跨模态交互关注信息传递，对齐损失进一步约束同一兴趣点的不同模态在共享空间中接近。设批次内共有 \(B\) 个兴趣点，模态 \(a\) 与模态 \(b\) 的 InfoNCE 损失为：

\[
\mathcal L_{a,b}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{
\exp(
\operatorname{sim}
(\widetilde{\boldsymbol e}_{i}^{a},
 \widetilde{\boldsymbol e}_{i}^{b})/\tau
)}
{
\sum_{j=1}^{B}
\exp(
\operatorname{sim}
(\widetilde{\boldsymbol e}_{i}^{a},
 \widetilde{\boldsymbol e}_{j}^{b})/\tau
)
},
\]

其中：

\[
\operatorname{sim}(\boldsymbol a,\boldsymbol b)
=
\frac{\boldsymbol a^\top\boldsymbol b}
{\|\boldsymbol a\|_2\|\boldsymbol b\|_2},
\]

\(\tau\) 为温度系数。

总跨模态对齐损失为：

\[
\mathcal L_{align}
=
\mathcal L_{sem,topo}
+
\mathcal L_{sem,spa}
+
\mathcal L_{topo,spa}.
\]

该损失仅用于表示学习和联合训练，不作为后续融合模块的显式输入。

---

# 4. 用户画像编码

## 4.1 画像统计

用户画像由训练集历史签到记录构建，主要包含四类统计先验：

### （1）类别偏好

\[
\boldsymbol \pi_u^{cat}[c]
=
\frac{
\sum_{p:c_p=c} n_{u,p}
}{
\sum_{p}n_{u,p}
},
\]

其中 \(n_{u,p}\) 表示用户 \(u\) 对兴趣点 \(p\) 的训练集访问权重或访问次数。

### （2）时间偏好

将一周划分为星期—小时槽或其他时间统计槽，得到：

\[
\boldsymbol \pi_u^{time}
\in\mathbb R^{D_t}.
\]

若采用 7 天 × 24 小时划分，则 \(D_t=168\)；当前代码也支持从预处理数据读入用户时间画像。

### （3）空间分布

根据用户历史访问位置计算空间中心与离散程度：

\[
\boldsymbol \mu_u
=
\frac{
\sum_p n_{u,p}\boldsymbol l_p
}{
\sum_p n_{u,p}
},
\]

\[
\boldsymbol \sigma_u
=
\sqrt{
\frac{
\sum_p n_{u,p}\boldsymbol l_p^{\odot 2}
}{
\sum_p n_{u,p}
}
-
\boldsymbol \mu_u^{\odot 2}
}.
\]

用户空间画像为：

\[
\boldsymbol \pi_u^{spa}
=
[\boldsymbol \mu_u;\boldsymbol \sigma_u].
\]

### （4）历史访问分布

保留用户历史访问兴趣点及其归一化权重：

\[
\mathcal H_u^{poi}
=
\{(p,w_{u,p})\},
\]

\[
\bar w_{u,p}
=
\frac{w_{u,p}}
{\sum_{p'}w_{u,p'}}.
\]

---

## 4.2 用户画像表示

将类别偏好、时间偏好、空间分布和历史访问分布分别编码：

\[
\boldsymbol l_u^{cat}
=
f_{cat}(\boldsymbol \pi_u^{cat}),
\]

\[
\boldsymbol l_u^{time}
=
f_{time}(\boldsymbol \pi_u^{time}),
\]

\[
\boldsymbol l_u^{spa}
=
f_{spa}(\boldsymbol \pi_u^{spa}),
\]

\[
\boldsymbol l_u^{poi}
=
\sum_{p\in\mathcal H_u^{poi}}
\bar w_{u,p}\boldsymbol E_p.
\]

最终用户画像表示为：

\[
\boldsymbol l_u
=
\operatorname{LN}
\left(
\operatorname{MLP}_{profile}
\left[
\boldsymbol l_u^{cat};
\boldsymbol l_u^{time};
\boldsymbol l_u^{spa};
\boldsymbol l_u^{poi}
\right]
\right).
\]

\(\boldsymbol l_u\) 同时承担两种作用：

1. 控制兴趣点多模态自适应融合；
2. 作为时空签到序列的用户条件前缀。

---

# 5. 上下文感知的自适应融合

对于不同用户，同一兴趣点的语义、拓扑和空间信息具有不同重要性。例如，通勤型用户可能更依赖转移拓扑，旅游型用户可能更关注语义类别和空间邻近性。因此，本文以用户画像为条件生成模态权重。

## 5.1 门控权重

设门控网络为：

\[
\boldsymbol g_u
=
\operatorname{MLP}_{gate}(\boldsymbol l_u).
\]

通过 Softmax 将其归一化为三个模态权重：

\[
[\alpha_u^{sem},
\alpha_u^{topo},
\alpha_u^{spa}]
=
\operatorname{softmax}(\boldsymbol g_u),
\]

满足：

\[
\alpha_u^{sem}
+
\alpha_u^{topo}
+
\alpha_u^{spa}
=1.
\]

实际实现也可以采用 sigmoid 门控；论文中应根据 `AdaptiveFusionGate` 的最终实现决定写 Softmax 还是 Sigmoid。若输出经 Softmax 归一化，则上述公式可直接使用。

## 5.2 融合兴趣点表示

兴趣点 \(p\) 面向用户 \(u\) 的融合表示为：

\[
\boldsymbol h_{u,p}
=
\alpha_u^{sem}
\widetilde{\boldsymbol e}_p^{sem}
+
\alpha_u^{topo}
\widetilde{\boldsymbol e}_p^{topo}
+
\alpha_u^{spa}
\widetilde{\boldsymbol e}_p^{spa}.
\]

为简化符号，在上下文明确时记为：

\[
\boldsymbol h_p
=
\alpha_u^{sem}
\widetilde{\boldsymbol e}_p^{sem}
+
\alpha_u^{topo}
\widetilde{\boldsymbol e}_p^{topo}
+
\alpha_u^{spa}
\widetilde{\boldsymbol e}_p^{spa}.
\]

对于用户历史序列中的每个位置 \(p_i\)，均以同一用户画像 \(\boldsymbol l_u\) 作为门控条件，生成对应的融合签到表示 \(\boldsymbol h_{p_i}\)。

---

# 6. 代价感知的自适应级联多索引检索

## 6.1 设计动机

固定执行全部检索源会产生两类问题：

1. 对容易样本执行复杂检索，造成不必要的计算和索引访问；
2. 不同检索源的质量和代价差异较大，简单合并难以在召回率与效率之间取得平衡。

本文将检索源划分为低代价集合和高代价集合：

\[
\mathcal R^{cheap}
=
\{r_1,r_2,\ldots,r_m\},
\]

\[
\mathcal R^{exp}
=
\{r_{m+1},\ldots,r_M\}.
\]

第一阶段始终执行低代价召回；仅当初始候选集覆盖不足或置信度较低时，第二阶段才执行高代价召回。

---

## 6.2 多索引检索源

结合当前数据组织和候选生成逻辑，检索源可归纳为：

| 检索源 | 主要依据 | 典型实现 | 相对代价 |
|---|---|---|---|
| 历史检索 | 用户近期访问 POI | 历史列表倒序衰减 | 低 |
| 转移检索 | 末次 POI 的转移邻居 | 转移图邻接表 | 低 |
| 历史转移检索 | 更早历史 POI 的转移邻居 | 多行邻接表 | 低—中 |
| 用户检索 | 用户历史高频 POI | 用户—POI 倒排表 | 低 |
| 时间检索 | 相同星期和小时模式 | 时间桶索引 | 低 |
| 类别检索 | 与末次 POI 相同类别 | 类别倒排索引 | 低 |
| 流行度检索 | 全局高频 POI | 排序列表 | 低 |
| 空间近邻 | 末次位置附近 POI | 坐标近邻检索 | 中—高 |
| 轨迹语义检索 | 相似轨迹上下文 | 轨迹条件索引 | 中—高 |
| 扩展检索 | 扩大索引范围或补充复杂检索 | 多索引扩展 | 高 |

论文图中可以将历史、转移、用户、时间、类别和流行度放在 Stage 1，将空间近邻、轨迹语义和扩展检索放在 Stage 2。具体分组应与最终本地实现保持一致。

---

## 6.3 单一检索源得分归一化

检索源 \(r\) 返回候选集合：

\[
\mathcal C_u^{r}
=
\{(p,\rho_r(u,p))\},
\]

其中 \(\rho_r(u,p)\) 为源内原始分数。

为消除不同检索源的量纲差异，对源内分数进行归一化：

\[
\bar\rho_r(u,p)
=
\frac{
\rho_r(u,p)
}{
\max_{p'\in \mathcal C_u^r}\rho_r(u,p')+\epsilon
}.
\]

对于距离型检索，可先将距离转换为相似度。例如：

\[
\bar\rho_{spa}(u,p)
=
1-
\frac{
d(p,p_{t-1})
}{
\max_{p'\in\mathcal C_u^{spa}}d(p',p_{t-1})+\epsilon
}.
\]

---

## 6.4 候选合并、去重和来源标记

Stage 1 的初始候选集合为：

\[
\mathcal C_u^{(1)}
=
\bigcup_{r\in\mathcal R^{cheap}}
\mathcal C_u^r.
\]

同一兴趣点可能被多个检索源召回。对候选 \(p\)，其融合检索分数为：

\[
R_u^{(1)}(p)
=
\sum_{r\in\mathcal R^{cheap}}
\beta_r
\bar\rho_r(u,p),
\]

其中 \(\beta_r\) 为检索源权重。

使用位掩码或多热向量记录候选来源：

\[
\boldsymbol f_u^{src}(p)
=
[f_{u,p}^{1},f_{u,p}^{2},\ldots,f_{u,p}^{M}],
\]

\[
f_{u,p}^{r}
=
\mathbb I[p\in\mathcal C_u^r].
\]

来源标记既可用于统计来源覆盖，也可作为后续重排序特征。

---

## 6.5 级联扩展判定

Stage 1 完成后，计算候选数量：

\[
n_u^{(1)}
=
|\mathcal C_u^{(1)}|.
\]

将候选分数归一化为概率分布：

\[
\pi_u^{(1)}(p)
=
\frac{
\exp(R_u^{(1)}(p)/\tau_r)
}{
\sum_{p'\in\mathcal C_u^{(1)}}
\exp(R_u^{(1)}(p')/\tau_r)
}.
\]

候选分数熵定义为：

\[
H_u^{(1)}
=
-\sum_{p\in\mathcal C_u^{(1)}}
\pi_u^{(1)}(p)
\log
\pi_u^{(1)}(p).
\]

为便于跨样本比较，可使用归一化熵：

\[
\bar H_u^{(1)}
=
\frac{
H_u^{(1)}
}{
\log(n_u^{(1)}+\epsilon)
}.
\]

扩展决策可采用以下三种规则。

### （1）数量规则

\[
g_u^{count}
=
\mathbb I
\left[
n_u^{(1)}<\theta_n
\right].
\]

### （2）熵规则

\[
g_u^{entropy}
=
\mathbb I
\left[
\bar H_u^{(1)}>\theta_H
\right].
\]

高熵表示候选分数较为平坦，Stage 1 缺少高置信度候选，需要扩展检索。

### （3）数量与熵联合规则

\[
g_u
=
\mathbb I
\left[
n_u^{(1)}<\theta_n
\ \lor\
\bar H_u^{(1)}>\theta_H
\right].
\]

其中：

- \(g_u=0\)：提前停止，直接输出 Stage 1 候选；
- \(g_u=1\)：执行 Stage 2 高代价召回。

也可以采用“与”规则：

\[
g_u
=
\mathbb I
\left[
n_u^{(1)}<\theta_n
\ \land\
\bar H_u^{(1)}>\theta_H
\right],
\]

但该规则更保守，通常会降低 Stage 2 调用比例。

---

## 6.6 Stage 2 高代价召回

当 \(g_u=1\) 时，执行高代价检索源：

\[
\mathcal C_u^{(2)}
=
\bigcup_{r\in\mathcal R^{exp}}
\mathcal C_u^r.
\]

最终合并集合为：

\[
\widetilde{\mathcal C}_u
=
\mathcal C_u^{(1)}
\cup
\mathcal C_u^{(2)}.
\]

更新候选融合分数：

\[
R_u(p)
=
R_u^{(1)}(p)
+
g_u
\sum_{r\in\mathcal R^{exp}}
\beta_r\bar\rho_r(u,p).
\]

当 \(g_u=0\) 时：

\[
\widetilde{\mathcal C}_u
=
\mathcal C_u^{(1)}.
\]

按照融合检索分数截断至候选预算 \(N\)：

\[
\mathcal C_u
=
\operatorname{TopN}_{p\in\widetilde{\mathcal C}_u}
R_u(p).
\]

---

## 6.7 检索效率目标

设各检索源成本为 \(c_r\)，则每个样本的实际检索成本为：

\[
C_u^{retr}
=
\sum_{r\in\mathcal R^{cheap}}c_r
+
g_u
\sum_{r\in\mathcal R^{exp}}c_r.
\]

平均检索成本为：

\[
\overline C^{retr}
=
\frac{1}{|\mathcal D|}
\sum_{u\in\mathcal D}
C_u^{retr}.
\]

Stage 2 调用率定义为：

\[
\mathrm{ExpandRate}
=
\frac{1}{|\mathcal D|}
\sum_{u\in\mathcal D}g_u.
\]

相对于固定执行全部检索源的成本节省比例为：

\[
\mathrm{Saving}
=
1-
\frac{
\sum_u C_u^{retr}
}{
|\mathcal D|
\sum_{r\in\mathcal R^{cheap}\cup\mathcal R^{exp}}c_r
}.
\]

该机制的目标是在维持 Recall@\(N\) 和最终推荐性能的同时，降低平均检索延迟及高代价索引访问次数。

---

# 7. 用户画像条件化的时空序列建模

## 7.1 历史融合兴趣点序列

对历史序列中第 \(i\) 个兴趣点 \(p_i\)，首先根据用户画像生成融合表示：

\[
\boldsymbol h_{p_i}
=
\alpha_u^{sem}
\widetilde{\boldsymbol e}_{p_i}^{sem}
+
\alpha_u^{topo}
\widetilde{\boldsymbol e}_{p_i}^{topo}
+
\alpha_u^{spa}
\widetilde{\boldsymbol e}_{p_i}^{spa}.
\]

由此得到历史融合兴趣点序列：

\[
\boldsymbol H_u^{t-1}
=
[
\boldsymbol h_{p_1},
\boldsymbol h_{p_2},
\ldots,
\boldsymbol h_{p_{t-1}}
].
\]

---

## 7.2 相对末次位置空间编码

当前代码以历史序列的末次兴趣点 \(p_{t-1}\) 作为空间参照。对历史兴趣点 \(p_i\)，计算：

\[
\Delta \mathrm{lat}_i
=
\mathrm{lat}_{p_i}
-
\mathrm{lat}_{p_{t-1}},
\]

\[
\Delta \mathrm{lon}_i
=
\mathrm{lon}_{p_i}
-
\mathrm{lon}_{p_{t-1}},
\]

\[
d_i
=
\sqrt{
(\Delta \mathrm{lat}_i)^2
+
(\Delta \mathrm{lon}_i)^2
}.
\]

相对空间特征为：

\[
\boldsymbol \delta_i^{spa}
=
[
\Delta \mathrm{lat}_i,
\Delta \mathrm{lon}_i,
d_i
].
\]

空间关系编码为：

\[
\boldsymbol r_i^{spa}
=
\operatorname{LN}
\left(
\operatorname{MLP}_{rel}
(\boldsymbol \delta_i^{spa})
\right).
\]

该设计显式刻画每个历史签到相对于用户当前轨迹末端的位置关系。

---

## 7.3 历史签到时间槽编码

将签到时间映射到一周中的星期—小时槽：

\[
s_i
=
24\cdot \mathrm{weekday}_i
+
\mathrm{hour}_i,
\qquad
s_i\in\{0,\ldots,167\}.
\]

时间槽编码为：

\[
\boldsymbol e_i^{time}
=
\boldsymbol E_{time}[s_i].
\]

将兴趣点融合表示、相对空间编码和历史时间槽编码相加，构造时空增强签到表示：

\[
\boldsymbol x_i
=
\boldsymbol h_{p_i}
+
\boldsymbol r_i^{spa}
+
\boldsymbol e_i^{time}.
\]

时空增强历史序列为：

\[
\boldsymbol X_u^{t-1}
=
[
\boldsymbol x_1,
\boldsymbol x_2,
\ldots,
\boldsymbol x_{t-1}
].
\]

---

## 7.4 用户画像前缀条件化

用户画像不是在 Transformer 输出后再次融合，而是作为序列前缀参与序列编码。条件序列为：

\[
\overline{\boldsymbol X}_u
=
[
\boldsymbol l_u,
\boldsymbol x_1,
\boldsymbol x_2,
\ldots,
\boldsymbol x_{t-1}
].
\]

这种设计使所有历史签到状态均能够在因果约束下感知用户长期偏好。

---

## 7.5 因果 Transformer 序列编码

将条件序列输入因果 Transformer：

\[
\boldsymbol Z_u
=
\operatorname{Transformer}_{causal}
(
\overline{\boldsymbol X}_u
).
\]

因果掩码定义为：

\[
M_{ij}
=
\begin{cases}
0, & j\le i,\\
-\infty, & j>i.
\end{cases}
\]

第 \(l\) 层自注意力可写为：

\[
\boldsymbol Q^{(l)}
=
\boldsymbol Z^{(l-1)}
\boldsymbol W_Q^{(l)},
\]

\[
\boldsymbol K^{(l)}
=
\boldsymbol Z^{(l-1)}
\boldsymbol W_K^{(l)},
\]

\[
\boldsymbol V^{(l)}
=
\boldsymbol Z^{(l-1)}
\boldsymbol W_V^{(l)},
\]

\[
\operatorname{CausalAttn}
=
\operatorname{softmax}
\left(
\frac{
\boldsymbol Q^{(l)}
(\boldsymbol K^{(l)})^\top
}{
\sqrt d
}
+
\boldsymbol M
\right)
\boldsymbol V^{(l)}.
\]

模型选取最后一个有效历史签到位置的输出作为末次签到上下文表示：

\[
\boldsymbol z_{t-1}
=
\boldsymbol Z_u[t].
\]

这里索引偏移一位，是因为序列首位为用户画像前缀 \(\boldsymbol l_u\)。

---

## 7.6 目标条件化查询构建

为了刻画“用户将在何时进行下一次签到”，在末次签到上下文表示上加入目标时间和签到间隔条件。

目标星期—小时槽为：

\[
s_t
=
24\cdot \mathrm{weekday}_t
+
\mathrm{hour}_t,
\]

目标时间编码为：

\[
\boldsymbol e_t^{target}
=
\boldsymbol E_{target}[s_t].
\]

将目标签到间隔 \(\Delta t_t\) 离散为区间编号 \(b_t\)：

\[
b_t
=
\operatorname{Bucket}(\Delta t_t),
\]

其间隔编码为：

\[
\boldsymbol e_t^{gap}
=
\boldsymbol E_{gap}[b_t].
\]

目标条件增强状态为：

\[
\widetilde{\boldsymbol z}_{t-1}
=
\boldsymbol z_{t-1}
+
\boldsymbol e_t^{target}
+
\boldsymbol e_t^{gap}.
\]

用户动态查询表示为：

\[
\boldsymbol q_u
=
\operatorname{LN}
\left(
\operatorname{MLP}_{query}
(
\widetilde{\boldsymbol z}_{t-1}
)
\right).
\]

因此：

\[
\boxed{
\boldsymbol q_u
=
\operatorname{MLP}_{query}
\left(
\boldsymbol z_{t-1}
+
\boldsymbol e_t^{target}
+
\boldsymbol e_t^{gap}
\right)
}
\]

需要特别强调：\(\boldsymbol l_u\) 已在 Transformer 输入端作为前缀参与建模，不应在此处与 \(\boldsymbol z_{t-1}\) 再次并列相加。

---

# 8. 候选重排序与下一兴趣点推荐

## 8.1 候选兴趣点表示

对于候选兴趣点 \(p\in\mathcal C_u\)，计算其融合表示 \(\boldsymbol h_p\)。在当前实现中，候选表示可以由三模态编码结果经过融合模块得到，并投影到与查询表示相同的隐空间：

\[
\boldsymbol v_p
=
\operatorname{Proj}_{cand}
(\boldsymbol h_p).
\]

为进行相似度匹配，对查询和候选表示进行 \(L_2\) 归一化：

\[
\bar{\boldsymbol q}_u
=
\frac{\boldsymbol q_u}{\|\boldsymbol q_u\|_2},
\]

\[
\bar{\boldsymbol v}_p
=
\frac{\boldsymbol v_p}{\|\boldsymbol v_p\|_2}.
\]

---

## 8.2 神经匹配得分

神经匹配得分为缩放余弦相似度：

\[
s_{match}(u,p)
=
\gamma
\bar{\boldsymbol q}_u^\top
\bar{\boldsymbol v}_p,
\]

其中：

\[
\gamma
=
\exp(\omega),
\]

\(\omega\) 为可学习参数，并在实现中设置最大值以保证训练稳定性。

---

## 8.3 检索分数与来源偏置

将多索引检索得到的归一化融合分数记为：

\[
R_u(p).
\]

其对应重排序偏置为：

\[
s_{retr}(u,p)
=
\lambda_r R_u(p),
\]

其中 \(\lambda_r\) 为可学习或预设权重。

候选来源标记为 \(\boldsymbol f_u^{src}(p)\)，来源偏置为：

\[
s_{src}(u,p)
=
\boldsymbol b_{src}^{\top}
\boldsymbol f_u^{src}(p),
\]

其中 \(\boldsymbol b_{src}\) 为可学习来源权重。

---

## 8.4 候选上下文特征打分

当前代码支持四类候选上下文特征：

1. 候选与末次签到位置之间的相对距离；
2. 候选类别与末次签到类别的一致性；
3. 候选融合检索分数；
4. 候选是否具有至少一个召回来源。

定义：

\[
d_{u,p}
=
\|
\boldsymbol l_p
-
\boldsymbol l_{p_{t-1}}
\|_2,
\]

\[
a_{u,p}^{cat}
=
\mathbb I[c_p=c_{p_{t-1}}],
\]

\[
a_{u,p}^{src}
=
\mathbb I[
\|\boldsymbol f_u^{src}(p)\|_1>0
].
\]

候选特征向量为：

\[
\boldsymbol f_{u,p}^{cand}
=
[
d_{u,p},
a_{u,p}^{cat},
R_u(p),
a_{u,p}^{src}
].
\]

特征得分为：

\[
s_{feat}(u,p)
=
\operatorname{MLP}_{feat}
(
\boldsymbol f_{u,p}^{cand}
).
\]

若候选特征 MLP 未启用，则可以令：

\[
s_{feat}(u,p)=0.
\]

---

## 8.5 最终得分

完整候选得分可写为：

\[
s(u,p)
=
s_{match}(u,p)
+
s_{retr}(u,p)
+
s_{src}(u,p)
+
s_{feat}(u,p).
\]

若论文图希望保持简洁，可以将检索分数、来源偏置和显式上下文特征统一归入 \(s_{feat}\)：

\[
s(u,p)
=
s_{match}(u,p)
+
s_{feat}(u,p).
\]

此时定义：

\[
s_{feat}(u,p)
=
\lambda_r R_u(p)
+
\boldsymbol b_{src}^{\top}
\boldsymbol f_u^{src}(p)
+
\operatorname{MLP}_{feat}
(
\boldsymbol f_{u,p}^{cand}
).
\]

最终返回：

\[
\hat{\mathcal P}_{u,t}^{K}
=
\operatorname{TopK}_{p\in\mathcal C_u}
s(u,p).
\]

---

# 9. 模型训练目标

## 9.1 候选排序损失

对于一个训练样本，设真实下一兴趣点在候选池中的索引为 \(y\)，候选得分为 \(\{s_j\}_{j=1}^{|\mathcal C_u|}\)，则排序交叉熵损失为：

\[
\mathcal L_{rank}
=
-
\log
\frac{
\exp(s_y)
}{
\sum_{j=1}^{|\mathcal C_u|}
\exp(s_j)
}.
\]

候选掩码用于排除填充位置。

---

## 9.2 跨模态对齐损失

跨模态对齐损失为：

\[
\mathcal L_{align}
=
\mathcal L_{sem,topo}
+
\mathcal L_{sem,spa}
+
\mathcal L_{topo,spa}.
\]

在总目标中的权重记为 \(\lambda_{align}\)。

---

## 9.3 辅助任务损失

当前实现可选地包含类别预测和地理簇预测等辅助任务。

### 类别预测损失

\[
\widehat{\boldsymbol y}_u^{cat}
=
\boldsymbol W_{cat}\boldsymbol q_u+\boldsymbol b_{cat},
\]

\[
\mathcal L_{cat}
=
\operatorname{CE}
(
\widehat{\boldsymbol y}_u^{cat},
c_{p_t}
).
\]

### 地理簇预测损失

设兴趣点地理簇标签为 \(g_{p_t}\)，则：

\[
\widehat{\boldsymbol y}_u^{geo}
=
\boldsymbol W_{geo}\boldsymbol q_u+\boldsymbol b_{geo},
\]

\[
\mathcal L_{geo}
=
\operatorname{CE}
(
\widehat{\boldsymbol y}_u^{geo},
g_{p_t}
).
\]

若主配置未启用对应模块，则论文主方法中不必写入这些辅助任务，可放在实现细节或扩展实验中。

---

## 9.4 总训练目标

总损失为：

\[
\mathcal L
=
\mathcal L_{rank}
+
\lambda_{align}\mathcal L_{align}
+
\lambda_{cat}\mathcal L_{cat}
+
\lambda_{geo}\mathcal L_{geo}
+
\lambda_{reg}\|\Theta\|_2^2.
\]

若只启用排序和对齐损失，则简化为：

\[
\boxed{
\mathcal L
=
\mathcal L_{rank}
+
\lambda_{align}\mathcal L_{align}
}
\]

---

# 10. 推理过程

对于每个用户查询，模型执行以下步骤：

1. 根据用户训练历史统计类别、时间、空间和访问分布，获得用户画像 \(\boldsymbol l_u\)；
2. 编码所有历史兴趣点的语义、拓扑和空间表示；
3. 进行跨模态交互，并使用用户画像门控融合得到 \(\boldsymbol h_{p_i}\)；
4. 执行 Stage 1 低代价多源召回；
5. 根据 \(n_u^{(1)}\) 和 \(\bar H_u^{(1)}\) 判断是否执行 Stage 2；
6. 合并、去重并截断得到动态候选池 \(\mathcal C_u\)；
7. 构建时空增强历史签到序列 \(\boldsymbol x_1,\ldots,\boldsymbol x_{t-1}\)；
8. 将用户画像作为前缀输入因果 Transformer；
9. 取末次有效签到状态，并加入目标时间和签到间隔编码，得到 \(\boldsymbol q_u\)；
10. 对 \(\mathcal C_u\) 中每个候选计算神经匹配分数及候选上下文得分；
11. 按最终得分排序并输出 Top-\(K\) 推荐结果。

---

# 11. 算法伪代码

```text
算法 1：多模态下一兴趣点推荐

输入：
    用户 u
    历史签到序列 S_u
    目标时间条件 (t_t, Δt_t)
    多模态兴趣点数据
输出：
    Top-K 下一兴趣点推荐结果

1:  l_u ← ProfileEncoder(u 的历史统计)
2:  for p_i in S_u do
3:      e_sem, e_topo, e_spa ← MultiModalEncoder(p_i)
4:      ẽ_sem, ẽ_topo, ẽ_spa ← CrossModalInteraction(...)
5:      α_sem, α_topo, α_spa ← GatingNetwork(l_u)
6:      h_{p_i} ← α_sem ẽ_sem + α_topo ẽ_topo + α_spa ẽ_spa
7:  end for

8:  C_u^(1), R_u^(1), F_u^(1) ← CheapMultiIndexRetrieve(u, S_u)
9:  n_u^(1) ← |C_u^(1)|
10: H_u^(1) ← ScoreEntropy(R_u^(1))
11: g_u ← CascadeDecision(n_u^(1), H_u^(1))

12: if g_u = 1 then
13:     C_u^(2), R_u^(2), F_u^(2) ← ExpensiveMultiIndexRetrieve(u, S_u)
14:     C_u ← MergeAndTopN(C_u^(1), C_u^(2))
15: else
16:     C_u ← TopN(C_u^(1))
17: end if

18: for each historical check-in i do
19:     r_i^spa ← RelativeSpatialEncoder(p_i, p_{t-1})
20:     e_i^time ← HistoricalTimeEmbedding(t_i)
21:     x_i ← h_{p_i} + r_i^spa + e_i^time
22: end for

23: Z_u ← CausalTransformer([l_u, x_1, ..., x_{t-1}])
24: z_{t-1} ← 最后有效历史签到位置的隐藏状态
25: q_u ← QueryProjection(z_{t-1} + TargetTime(t_t) + GapEmbedding(Δt_t))

26: for p in C_u do
27:     h_p ← CandidateMultimodalRepresentation(p)
28:     s_match ← ScaledCosine(q_u, h_p)
29:     s_feat ← CandidateFeatureScore(p, R_u(p), F_u(p))
30:     s(u,p) ← s_match + s_feat
31: end for

32: return TopK_{p∈C_u} s(u,p)
```

---

# 12. 复杂度分析

设：

- 历史序列长度为 \(L\)；
- 隐空间维度为 \(d\)；
- 动态候选池大小为 \(N\)；
- 因果 Transformer 层数为 \(L_T\)；
- Stage 1 和 Stage 2 检索成本分别为 \(C_{cheap}\) 和 \(C_{exp}\)。

## 12.1 时空序列建模复杂度

Transformer 序列编码复杂度约为：

\[
O(L_TL^2d).
\]

多模态编码与融合复杂度约为：

\[
O(Ld^2).
\]

## 12.2 候选重排序复杂度

对 \(N\) 个候选进行查询—候选匹配的复杂度为：

\[
O(Nd).
\]

当 \(N\ll|\mathcal P|\) 时，相比全兴趣点打分：

\[
O(|\mathcal P|d),
\]

动态候选重排序可显著降低推理成本。

## 12.3 自适应级联检索复杂度

单样本期望检索成本为：

\[
\mathbb E[C^{retr}]
=
C_{cheap}
+
P(g_u=1)C_{exp}.
\]

固定执行全部检索源的成本为：

\[
C_{fixed}
=
C_{cheap}+C_{exp}.
\]

只要：

\[
P(g_u=1)<1,
\]

且 Stage 1 对容易样本具有足够覆盖，自适应级联即可减少平均检索成本。

---

# 13. 主要符号表

| 符号 | 含义 |
|---|---|
| \(\mathcal U\) | 用户集合 |
| \(\mathcal P\) | 兴趣点集合 |
| \(\mathcal S_u^{t-1}\) | 用户 \(u\) 在时刻 \(t\) 前的历史签到序列 |
| \(p_i\) | 历史序列中第 \(i\) 个兴趣点 |
| \(p_t\) | 真实下一兴趣点 |
| \(c_p\) | 兴趣点类别 |
| \(\boldsymbol l_p\) | 兴趣点地理坐标 |
| \(\boldsymbol e_p^{sem}\) | 原始语义模态表示 |
| \(\boldsymbol e_p^{topo}\) | 原始拓扑模态表示 |
| \(\boldsymbol e_p^{spa}\) | 原始空间模态表示 |
| \(\widetilde{\boldsymbol e}_p^{sem}\) | 交互后的语义表示 |
| \(\widetilde{\boldsymbol e}_p^{topo}\) | 交互后的拓扑表示 |
| \(\widetilde{\boldsymbol e}_p^{spa}\) | 交互后的空间表示 |
| \(\boldsymbol l_u\) | 用户画像表示 |
| \(\alpha_u^{sem}\) | 用户对语义模态的门控权重 |
| \(\alpha_u^{topo}\) | 用户对拓扑模态的门控权重 |
| \(\alpha_u^{spa}\) | 用户对空间模态的门控权重 |
| \(\boldsymbol h_p\) | 融合兴趣点表示 |
| \(\boldsymbol r_i^{spa}\) | 相对末次位置空间编码 |
| \(\boldsymbol e_i^{time}\) | 历史签到时间槽编码 |
| \(\boldsymbol x_i\) | 时空增强签到表示 |
| \(\boldsymbol z_{t-1}\) | 末次有效签到上下文表示 |
| \(\boldsymbol e_t^{target}\) | 目标签到时间编码 |
| \(\boldsymbol e_t^{gap}\) | 目标签到间隔编码 |
| \(\boldsymbol q_u\) | 用户动态查询表示 |
| \(\mathcal C_u^r\) | 检索源 \(r\) 的召回结果 |
| \(\mathcal C_u^{(1)}\) | Stage 1 初始候选集 |
| \(\mathcal C_u^{(2)}\) | Stage 2 扩展候选集 |
| \(\mathcal C_u\) | 最终动态候选池 |
| \(R_u(p)\) | 候选融合检索分数 |
| \(\boldsymbol f_u^{src}(p)\) | 候选来源标记 |
| \(n_u^{(1)}\) | Stage 1 候选数量 |
| \(H_u^{(1)}\) | Stage 1 候选分数熵 |
| \(g_u\) | 是否执行 Stage 2 的二元变量 |
| \(s_{match}(u,p)\) | 神经匹配得分 |
| \(s_{feat}(u,p)\) | 候选上下文特征得分 |
| \(s(u,p)\) | 最终推荐得分 |
| \(\mathcal L_{rank}\) | 候选排序损失 |
| \(\mathcal L_{align}\) | 跨模态对齐损失 |
| \(\tau\) | 对比学习温度系数 |
| \(d\) | 隐表示维度 |
| \(N\) | 动态候选池预算 |

---

# 14. 撰写论文时需要统一的关键表述

## 14.1 建议使用的模块名称

- 多模态表示学习与跨模态对齐
- 上下文感知的自适应融合
- 代价感知的自适应级联多索引检索
- 用户画像条件化的时空序列建模
- 候选重排序与下一兴趣点推荐

## 14.2 不建议使用的表述

1. 不要将当前序列编码器写成 GPT-2，除非后续重新接入 `GPTBackbone`；
2. 不要写“用户画像在 Transformer 之后与最终状态融合”；
3. 不要写“所有样本均执行全部召回源”；
4. 不要把动态候选池作为 Transformer 输入；
5. 不要将跨模态对齐损失画成或写成融合模块的输入；
6. 不要将相对空间编码笼统写成绝对空间位置编码；
7. 不要将历史时间编码和目标时间编码混为一类；
8. 若当前主配置未启用分时段层次聚合，不要将日级、时段级和签到级联合预测写入核心模型。

## 14.3 建议突出的方法贡献

1. **跨模态兴趣点表示**：联合建模语义、拓扑和空间信息，并通过跨模态交互及对齐损失减小模态异质性；
2. **画像条件化融合与序列建模**：用户画像既控制兴趣点多模态融合，又作为前缀条件进入因果时空序列编码；
3. **代价感知自适应检索**：根据 Stage 1 候选数量和分数熵动态选择是否访问高代价索引，在召回质量和检索效率之间取得平衡；
4. **检索—排序协同**：将检索分数、召回来源和候选上下文纳入最终排序，实现多索引召回与神经推荐模型的协同优化。

---

# 15. 与当前代码进一步核对的项目

正式写入论文前，建议逐项确认：

- `fusion.py` 中门控权重究竟采用 Softmax 还是 Sigmoid；
- 跨模态对齐模块使用的具体正负样本构造和温度参数；
- 当前实验配置中 `fusion.type` 是否固定为 `cross_modal`；
- `candidate_feature_mlp.enabled` 是否在主实验中启用；
- `target_time.enabled` 是否在所有主结果配置中启用；
- 类别辅助任务和地理簇辅助任务是否纳入最终主模型；
- 本地 CACR 实现中 cheap/expensive 检索源的最终划分；
- 级联判断采用“数量”“熵”还是“数量+熵”；
- Stage 1 熵使用原始熵还是归一化熵；
- Stage 2 的实际调用率、延迟和候选召回提升；
- 候选池固定预算 \(N\) 及各检索源的 Top-\(k\) 参数；
- CACR 是否已合并到论文实验所使用的代码版本。

---

## 16. 可直接作为方法章节的结构建议

```text
3 方法
3.1 问题定义与总体框架
3.2 多模态兴趣点表示学习
    3.2.1 语义、拓扑与空间编码
    3.2.2 跨模态交互
    3.2.3 跨模态对齐目标
3.3 用户画像编码与上下文感知自适应融合
    3.3.1 用户画像构建
    3.3.2 画像条件化模态门控
3.4 代价感知的自适应级联多索引检索
    3.4.1 多源索引与分数融合
    3.4.2 基于候选数量和分数熵的级联决策
    3.4.3 动态候选池构建
3.5 用户画像条件化的时空序列建模
    3.5.1 时空增强签到表示
    3.5.2 用户画像前缀条件化
    3.5.3 目标时间条件化查询
3.6 候选重排序与模型优化
    3.6.1 神经匹配
    3.6.2 候选上下文特征打分
    3.6.3 联合训练目标
```
