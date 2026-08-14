#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/vlm-rag"
cd "$PROJECT_ROOT"

export HF_HOME="/root/autodl-tmp/cache"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

BASELINE_RESULTS="outputs/siglip_chartqa_large_baseline/retrieval_results.json"
BASELINE_METRICS="outputs/siglip_chartqa_large_baseline/metrics.json"
STANDARD_METRICS="outputs/siglip_partial_full/metrics.json"

for required in "$BASELINE_RESULTS" "$BASELINE_METRICS" "$STANDARD_METRICS"; do
  if [[ ! -f "$required" ]]; then
    echo "Required previous experiment output is missing: $required" >&2
    exit 1
  fi
done

mkdir -p outputs/stage3_logs outputs/stage3_comparison

run_experiment() {
  local negative_count="$1"
  local batch_size="$2"
  local accumulation="$3"
  local output_dir="outputs/siglip_partial_hardneg_${negative_count}"
  local log_path="outputs/stage3_logs/hardneg_${negative_count}.log"

  if [[ -f "$output_dir/metrics.json" ]]; then
    echo "Skipping completed experiment: $output_dir"
    return
  fi

  echo "Starting hard-negative experiment: negatives=$negative_count"
  python scripts/train_siglip_partial_finetune.py \
    --dataset-dir data/real/chartqa_large \
    --output-dir "$output_dir" \
    --epochs 4 \
    --batch-size "$batch_size" \
    --eval-batch-size 32 \
    --gradient-accumulation "$accumulation" \
    --learning-rate 0.000002 \
    --weight-decay 0.01 \
    --temperature 0.07 \
    --warmup-ratio 0.05 \
    --unfreeze-text-layers 2 \
    --unfreeze-vision-layers 2 \
    --patience 2 \
    --num-workers 4 \
    --hard-negative-results "$BASELINE_RESULTS" \
    --hard-negatives-per-query "$negative_count" \
    2>&1 | tee "$log_path"
}

run_experiment 1 12 3
run_experiment 2 8 4

python scripts/summarize_stage3_experiments.py \
  --baseline "$BASELINE_METRICS" \
  --standard "$STANDARD_METRICS" \
  --hardneg-1 outputs/siglip_partial_hardneg_1/metrics.json \
  --hardneg-2 outputs/siglip_partial_hardneg_2/metrics.json \
  --output-dir outputs/stage3_comparison

tar -czf /root/autodl-tmp/stage3_complete_results.tar.gz \
  outputs/siglip_chartqa_large_baseline \
  outputs/siglip_partial_full \
  outputs/siglip_partial_hardneg_1 \
  outputs/siglip_partial_hardneg_2 \
  outputs/stage3_logs \
  outputs/stage3_comparison

echo "All Stage 3 experiments completed."
echo "Archive: /root/autodl-tmp/stage3_complete_results.tar.gz"
