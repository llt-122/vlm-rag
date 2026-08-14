#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/root/autodl-tmp/vlm-rag"
cd "$PROJECT_ROOT" || exit 1

export HF_HOME="/root/autodl-tmp/cache"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs/stage3_suite_logs outputs/stage3_large_baseline_comparison
STATUS_FILE="outputs/stage3_suite_logs/status.tsv"
printf "stage\tstatus\n" > "$STATUS_FILE"

run_stage() {
  local name="$1"
  shift
  local log="outputs/stage3_suite_logs/${name}.log"
  echo "===== START $name ====="
  set +e
  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -eq 0 ]]; then
    printf "%s\tcompleted\n" "$name" >> "$STATUS_FILE"
    echo "===== COMPLETE $name ====="
  else
    printf "%s\tfailed_%s\n" "$name" "$status" >> "$STATUS_FILE"
    echo "===== FAILED $name (exit $status); continuing =====" >&2
  fi
  return 0
}

if [[ ! -f data/real/chartqa_large/manifest.json ]]; then
  echo "Missing data/real/chartqa_large/manifest.json" >&2
  exit 1
fi
if [[ ! -f outputs/siglip_partial_full/siglip_partial_finetune.pt ]]; then
  echo "Missing best SigLIP checkpoint" >&2
  exit 1
fi

if [[ -f outputs/siglip_partial_chartqa_large/metrics.json ]]; then
  printf "tuned_siglip_index\tskipped_existing\n" >> "$STATUS_FILE"
  echo "Skipping existing tuned SigLIP index and metrics."
else
  run_stage tuned_siglip_index \
    python scripts/evaluate_siglip_partial_checkpoint.py \
      --dataset-dir data/real/chartqa_large \
      --checkpoint outputs/siglip_partial_full/siglip_partial_finetune.pt \
      --index-dir indexes/siglip_partial_chartqa_large \
      --output-dir outputs/siglip_partial_chartqa_large \
      --batch-size 32
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
run_stage install_ocr_dependencies \
  python -m pip install -r requirements-ocr.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --progress-bar off

run_stage extract_ppocr \
  python scripts/extract_chartqa_ocr.py \
    --dataset-dir data/real/chartqa_large \
    --output data/real/chartqa_large/ocr_text.jsonl \
    --minimum-score 0.5

export HF_ENDPOINT="https://hf-mirror.com"
[[ -f /etc/network_turbo ]] && source /etc/network_turbo || true
if [[ -f outputs/ocr_bge_chartqa_large/metrics.json ]]; then
  printf "evaluate_ocr_bge\tskipped_existing\n" >> "$STATUS_FILE"
  echo "Skipping existing OCR+BGE metrics."
else
  run_stage evaluate_ocr_bge \
    python scripts/evaluate_ocr_retrieval.py \
      --dataset-dir data/real/chartqa_large \
      --ocr-file data/real/chartqa_large/ocr_text.jsonl \
      --index-dir indexes/ocr_bge_chartqa_large \
      --output-dir outputs/ocr_bge_chartqa_large \
      --model BAAI/bge-small-en-v1.5 \
      --batch-size 64 \
      --device auto
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
run_stage install_colsmol_dependencies \
  python -m pip install -r requirements-colpali.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --progress-bar off

export HF_ENDPOINT="https://hf-mirror.com"
[[ -f /etc/network_turbo ]] && source /etc/network_turbo || true
if [[ -f outputs/colsmol_chartqa_large/metrics.json ]]; then
  printf "evaluate_colsmol\tskipped_existing\n" >> "$STATUS_FILE"
  echo "Skipping existing ColSmol metrics."
else
  run_stage evaluate_colsmol \
    python scripts/evaluate_colsmol_retrieval.py \
      --dataset-dir data/real/chartqa_large \
      --index-dir indexes/colsmol_chartqa_large \
      --output-dir outputs/colsmol_chartqa_large \
      --model vidore/colSmol-500M \
      --image-batch-size 2 \
      --query-batch-size 16 \
      --rebuild-index
fi

run_stage summarize_large_baselines \
  python scripts/summarize_large_retrieval_baselines.py \
    --output-dir outputs/stage3_large_baseline_comparison \
    --method siglip_baseline outputs/siglip_chartqa_large_baseline/metrics.json \
    --method siglip_partial_finetune outputs/siglip_partial_chartqa_large/metrics.json \
    --method ocr_bge outputs/ocr_bge_chartqa_large/metrics.json \
    --method colsmol outputs/colsmol_chartqa_large/metrics.json

archive_items=(
  outputs/siglip_chartqa_large_baseline
  outputs/siglip_partial_full
  outputs/siglip_partial_hardneg_1
  outputs/siglip_partial_hardneg_2
  outputs/siglip_partial_chartqa_large
  outputs/stage3_comparison
  outputs/stage3_large_baseline_comparison
  outputs/stage3_suite_logs
)
for optional in outputs/ocr_bge_chartqa_large outputs/colsmol_chartqa_large; do
  [[ -d "$optional" ]] && archive_items+=("$optional")
done
tar -czf /root/autodl-tmp/stage3_all_retrieval_results.tar.gz "${archive_items[@]}"

echo "===== PIPELINE FINISHED ====="
cat "$STATUS_FILE"
echo "Archive: /root/autodl-tmp/stage3_all_retrieval_results.tar.gz"
echo "The script does not shut down AutoDL automatically."
