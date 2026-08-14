#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/vlm-rag"
cd "$PROJECT_ROOT"

export HF_HOME="/root/autodl-tmp/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOCAL_MODEL="/root/autodl-tmp/models/SmolVLM-500M-Instruct"
if [[ -s "$LOCAL_MODEL/model.safetensors" ]]; then
  MODEL="$LOCAL_MODEL"
else
  MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
fi

DATASET="data/real/chartqa_large"
SIGLIP_RESULTS="outputs/siglip_partial_chartqa_large/retrieval_results.json"
COLSMOL_RESULTS="outputs/colsmol_chartqa_large/retrieval_results.json"
OUTPUT_DIR="outputs/stage4_generation_test"

for required in \
  "$DATASET/manifest.json" \
  "$SIGLIP_RESULTS" \
  "$COLSMOL_RESULTS"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required")
print("GPU:", torch.cuda.get_device_name(0))
PY

COMMON_ARGS=(
  --dataset-dir "$DATASET"
  --retriever "siglip_partial=$SIGLIP_RESULTS"
  --retriever "colsmol=$COLSMOL_RESULTS"
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL"
  --split test
  --top-k 3
  --max-new-tokens 16
  --collage-retriever colsmol
)

echo "===== STAGE 4 SMOKE TEST: 10 TEST QUERIES ====="
python scripts/evaluate_stage4_generation.py "${COMMON_ARGS[@]}" --limit 10

echo "===== STAGE 4 FULL TEST: 500 TEST QUERIES ====="
python scripts/evaluate_stage4_generation.py "${COMMON_ARGS[@]}"

tar -czf /root/autodl-tmp/stage4_generation_results.tar.gz \
  "$OUTPUT_DIR"

echo "===== STAGE 4 COMPLETE ====="
cat "$OUTPUT_DIR/comparison.md"
echo "Archive: /root/autodl-tmp/stage4_generation_results.tar.gz"
echo "The script does not shut down AutoDL automatically."
