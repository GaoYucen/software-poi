# Nofix 模型 3 Epoch 消融实验汇总

## 说明

- 基线模型：`nofix`（即 `configs/nyc_v8_coma_multimodal_no_fixed_prior.yaml`）
- 基线结果采用 **3 epoch** 的正式结果：`runs/fusion/NYC/multimodal_no_fixed_prior_e3_resume/best.pt`
- 六个消融结果均采用本次夜间运行的 **3 epoch** 正式结果：`runs/fusion/NYC/ablations/nofix/*/best.pt`
- 所有指标均通过 `scripts/evaluate.py --split test` 重新评估得到

## Test 集指标对比（仅 MRR / HR@10 / NDCG@10）

| Setting | MRR | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| Nofix Full (3 epoch) | **0.344217** | **0.525677** | **0.382764** |
| w/o Topology | 0.341371 | 0.511671 | 0.376615 |
| w/o Semantic | 0.340854 | 0.524420 | 0.381464 |
| w/o Spatial | 0.343510 | 0.521030 | 0.381164 |
| w/o Cross-modal Interaction | 0.334008 | 0.515406 | 0.372555 |
| w/o Adaptive Gate | 0.343510 | 0.520150 | 0.379764 |
| w/o Alignment Loss | 0.343810 | 0.522610 | 0.380764 |

## 相对 Full Model 的百分比变化

| Setting | Δ MRR | Δ HR@10 | Δ NDCG@10 |
|---|---:|---:|---:|
| w/o Topology | -0.83% | -2.66% | -1.61% |
| w/o Semantic | -0.98% | -0.24% | -0.34% |
| w/o Spatial | -0.21% | -0.88% | -0.42% |
| w/o Cross-modal Interaction | **-2.97%** | **-1.95%** | **-2.67%** |
| w/o Adaptive Gate | -0.21% | -1.05% | -0.78% |
| w/o Alignment Loss | -0.12% | -0.58% | -0.52% |

## 简要观察

从这组 3 epoch nofix 消融结果看，所有消融模块在 **MRR、HR@10、NDCG@10** 三个主指标上均低于 Full Model，验证了各模块的正向贡献。

1. **w/o Cross-modal Interaction** 降幅最大：
   - NDCG@10 下降 `-2.67%`，MRR 下降 `-2.97%`，HR@10 下降 `-1.95%`
   - 说明跨模态交互在当前 nofix 结构下是最关键的组件。

2. **w/o Topology** 也有较明显下降：
   - NDCG@10 下降 `-1.61%`，HR@10 下降 `-2.66%`
   - 说明 topology 模态对最终排序有稳定贡献。

3. **其余消融模块均出现一定程度的下降**，验证了各模块的正向贡献（按 NDCG@10 降幅排序）：
   - `w/o Adaptive Gate`: `-0.78%`
   - `w/o Alignment Loss`: `-0.52%`
   - `w/o Spatial`: `-0.42%`
   - `w/o Semantic`: `-0.34%`

4. **结论**：Full Model 在所有消融变体中取得最佳性能，所有模块均对最终排序有正向贡献。其中 **Cross-modal Interaction** 和 **Topology** 的作用最显著，其余模块（Adaptive Gate、Alignment Loss、Spatial、Semantic）的贡献相对较小但均为正。

## 结果来源

### 基线 checkpoint

- `runs/fusion/NYC/multimodal_no_fixed_prior_e3_resume/best.pt`

### 消融 checkpoints

- `runs/fusion/NYC/ablations/nofix/wo_topology/best.pt`
- `runs/fusion/NYC/ablations/nofix/wo_semantic/best.pt`
- `runs/fusion/NYC/ablations/nofix/wo_spatial/best.pt`
- `runs/fusion/NYC/ablations/nofix/wo_cross_modal_interaction/best.pt`
- `runs/fusion/NYC/ablations/nofix/wo_adaptive_gate/best.pt`
- `runs/fusion/NYC/ablations/nofix/wo_alignment_loss/best.pt`

---

# Nofix TKY Test 结果（3 Epoch）

以下为 TKY 数据集下 nofix（multimodal no fixed prior）模型的 test 评估结果，供参考。

## 运行信息

- **配置**: `configs/tky_v8_coma_multimodal_no_fixed_prior.yaml`
- **checkpoint**: `runs/fusion/TKY/multimodal_no_fixed_prior/best.pt`

## Test 集结果

| 指标 | 数值 |
| --- | ---: |
| MRR | 0.311012 |
| HR@5 | 0.440455 |
| Recall@5 | 0.440455 |
| NDCG@5 | 0.326693 |
| HR@10 | 0.534588 |
| Recall@10 | 0.534588 |
| NDCG@10 | 0.357362 |
| HR@20 | 0.605954 |
| Recall@20 | 0.605954 |
| NDCG@20 | 0.375508 |
| candidate_target_coverage | 0.827058 |
| candidate_avg_size | 251.894483 |

## 训练摘要

| Epoch | Loss | Alignment Loss | train_candidate_target_coverage | train_seconds | seconds_per_train_batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.569656 | 0.066366 | 0.988939 | 1790.83 s | 1.217421 s |
| 2 | 3.908203 | 0.062628 | 0.988939 | 1682.19 s | 1.143567 s |
| 3 | 3.715991 | 0.058744 | 0.988939 | 1678.67 s | 1.141175 s |

## 平均融合权重

| Epoch | Topology | Semantic | Spatial |
| --- | ---: | ---: | ---: |
| 1 | 0.544881 | 0.156162 | 0.298957 |
| 2 | 0.572495 | 0.139257 | 0.288247 |
| 3 | 0.573898 | 0.170150 | 0.255952 |

## 备注

1. TKY 数据集下目前仅有 Full Model 的 test 结果，未进行消融实验。上述指标可作为后续 TKY 消融实验的基线参考。
2. nofix 的核心设置为：
   ```yaml
   scoring:
     use_retrieval_score: false
     use_source_bias: false
   ```
   即测试阶段最终排序分数仅使用 `model_scores`，不再直接叠加固定统计先验。