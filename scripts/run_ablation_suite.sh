#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 CONFIG CITY GPU ALIGNMENT_CHECKPOINT" >&2
  exit 2
fi

config=$1
city=$2
gpu=$3
alignment_checkpoint=$4
root="runs/paper_revision/${city}/ablations"

variants=(
  "no_topology use_topology_modality=false align_loss_weight=0"
  "no_semantic use_semantic_modality=false align_loss_weight=0"
  "no_spatial use_spatial_modality=false"
  "no_temporal use_temporal_modality=false"
  "no_alignment alignment_mode=identity align_loss_weight=0"
  "instance_only align_instance_weight=1 align_feature_weight=0"
  "feature_only align_instance_weight=0 align_feature_weight=1"
  "concat fusion_mode=concat"
  "fixed_mean fusion_mode=fixed_mean"
  "gpt_full gpt_freeze_policy=full"
  "gpt_frozen gpt_freeze_policy=frozen"
)

for item in "${variants[@]}"; do
  read -r -a fields <<< "$item"
  name=${fields[0]}
  run_dir="${root}/${name}"
  overrides=(--override "run_dir=${run_dir}" --override "seed=42" --override "alignment_checkpoint=${alignment_checkpoint}")
  for ((index=1; index<${#fields[@]}; index++)); do
    overrides+=(--override "${fields[index]}")
  done
  CUDA_VISIBLE_DEVICES="$gpu" /opt/conda/envs/py11/bin/python scripts/train.py --config "$config" "${overrides[@]}"
  CUDA_VISIBLE_DEVICES="$gpu" /opt/conda/envs/py11/bin/python scripts/evaluate.py \
    --checkpoint "${run_dir}/best.pt" --split test > "${run_dir}/test_metrics.txt"
done
