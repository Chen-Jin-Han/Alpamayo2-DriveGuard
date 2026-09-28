#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
cd "$ROOT"
export PYTHONPATH=src
PYTHON=.venv-lite/bin/python
DATA=data/nuscenes
BASE=(--dataroot "$DATA" --samples 100 --frames 4 --sampling-strategy round_robin
      --resize-height 112 --resize-width 192 --iterations 3 --step-size 4 --threads 4)

run_stage() {
  local label="$1"
  shift
  local output="outputs/nuscenes_${label}"
  if [[ -s "$output/summary.json" ]]; then
    echo "stage_skip label=$label"
    return
  fi
  echo "stage_start label=$label"
  "$PYTHON" scripts/run_nuscenes_prototype.py "${BASE[@]}" --output-dir "$output" "$@"
  test -s "$output/summary.json"
  echo "stage_complete label=$label"
}

# Re-open the completed seed-42 rows to regenerate its summary with CI fields.
"$PYTHON" scripts/run_nuscenes_prototype.py "${BASE[@]}" \
  --attacks random,fgsm,pgd,temporal --epsilon-values 16 \
  --seed 42 --output-dir outputs/nuscenes_prototype_balanced_100

# Three-seed main comparison.
run_stage prototype_balanced_seed43 --attacks random,fgsm,pgd,temporal --epsilon-values 16 --seed 43
run_stage prototype_balanced_seed44 --attacks random,fgsm,pgd,temporal --epsilon-values 16 --seed 44

# Epsilon, camera-count, temporal-length, frame-mask, and objective ablations.
run_stage temporal_epsilon --attacks temporal --epsilon-values 2,4,8,16 --seed 42
run_stage temporal_cam1 --attacks temporal --epsilon-values 16 --attacked-cameras 1 --seed 42
run_stage temporal_cam3 --attacks temporal --epsilon-values 16 --attacked-cameras 0,1,2 --seed 42
run_stage temporal_t1 --attacks temporal --epsilon-values 16 --frames 1 --seed 42
run_stage temporal_t2 --attacks temporal --epsilon-values 16 --frames 2 --seed 42
run_stage temporal_last1 --attacks temporal --epsilon-values 16 --attacked-frames 3 --seed 42
run_stage temporal_last2 --attacks temporal --epsilon-values 16 --attacked-frames 2,3 --seed 42
run_stage objective_feature --attacks temporal --epsilon-values 16 \
  --objective-feature 1 --objective-trajectory 0 --objective-temporal 0 --seed 42
run_stage objective_trajectory --attacks temporal --epsilon-values 16 \
  --objective-feature 0 --objective-trajectory 1 --objective-temporal 0 --seed 42
run_stage objective_feature_trajectory --attacks temporal --epsilon-values 16 \
  --objective-feature 1 --objective-trajectory 1 --objective-temporal 0 --seed 42
run_stage objective_feature_temporal --attacks temporal --epsilon-values 16 \
  --objective-feature 1 --objective-trajectory 0 --objective-temporal 0.25 --seed 42
run_stage objective_full --attacks temporal --epsilon-values 16 \
  --objective-feature 1 --objective-trajectory 1 --objective-temporal 0.25 --seed 42
run_stage btc_shared --attacks temporal --epsilon-values 16 --shared-across-cameras --seed 42

"$PYTHON" scripts/aggregate_nuscenes_prototypes.py \
  --outputs outputs --destination outputs/nuscenes_study

echo "nuscenes_ablation_suite=completed"
