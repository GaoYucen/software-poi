#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CHECKPOINT CITY GPU" >&2
  exit 2
fi

checkpoint=$1
city=$2
gpu=$3
output_dir="runs/paper_revision/${city}/prior_ablation"
mkdir -p "$output_dir"

common=(
  --override transition_prior_weight=0
  --override history_transition_prior_weight=0
  --override user_poi_prior_weight=0
  --override co_visit_prior_weight=0
  --override category_transition_prior_weight=0
  --override user_category_prior_weight=0
  --override history_repeat_prior_weight=0
  --override popularity_prior_weight=0
  --override spatial_prior_weight=0
)

run_group() {
  local name=$1
  shift
  CUDA_VISIBLE_DEVICES="$gpu" /opt/conda/envs/py11/bin/python scripts/evaluate.py \
    --checkpoint "$checkpoint" --split test "${common[@]}" "$@" > "${output_dir}/${name}.txt"
}

run_group none
run_group transition \
  --override transition_prior_weight=8 --override history_transition_prior_weight=4
run_group user_repeat \
  --override user_poi_prior_weight=4 --override co_visit_prior_weight=0.5 \
  --override user_category_prior_weight=1 --override history_repeat_prior_weight=2
run_group popularity_spatial \
  --override popularity_prior_weight=0.3 --override spatial_prior_weight=0.5 \
  --override spatial_prior_temperature=0.05
run_group all \
  --override transition_prior_weight=8 --override history_transition_prior_weight=4 \
  --override user_poi_prior_weight=4 --override co_visit_prior_weight=0.5 \
  --override category_transition_prior_weight=1 --override user_category_prior_weight=1 \
  --override history_repeat_prior_weight=2 --override popularity_prior_weight=0.3 \
  --override spatial_prior_weight=0.5 --override spatial_prior_temperature=0.05
