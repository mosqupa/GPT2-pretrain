#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT="${1:-data/raw/corpus.txt}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT_VALUE:-https://hf-mirror.com}"
HF_TIMEOUT="${HF_TIMEOUT:-120}"
HF_RETRIES="${HF_RETRIES:-12}"
SUCCEEDED=0
FAILED=0

run_fetch() {
  local title="$1"
  shift
  echo "$title"
  if python3 scripts/fetch_corpus_hf.py "$@"; then
    SUCCEEDED=$((SUCCEEDED + 1))
  else
    FAILED=$((FAILED + 1))
    echo "[warn] 该数据源拉取失败，已跳过并继续下一个。"
  fi
}

run_fetch "[1/4] 追加 0xDing 中文维基（高质量，已验证可达）..." \
  --dataset 0xDing/wikipedia-cn-20230720-filtered \
  --text-column completion \
  --output "$OUTPUT" \
  --append \
  --min-length 20 \
  --max-lines 1200000 \
  --hf-endpoint "$HF_ENDPOINT_VALUE" \
  --hf-timeout "$HF_TIMEOUT" \
  --hf-retries "$HF_RETRIES"

run_fetch "[2/4] 追加 mC4 中文（超大规模网页语料）..." \
  --dataset mc4 \
  --config zh \
  --text-column text \
  --output "$OUTPUT" \
  --append \
  --min-length 20 \
  --max-lines 2000000 \
  --hf-endpoint "$HF_ENDPOINT_VALUE" \
  --hf-timeout "$HF_TIMEOUT" \
  --hf-retries "$HF_RETRIES"

run_fetch "[3/4] 追加 OSCAR 中文（网页语料）..." \
  --dataset oscar-corpus/OSCAR-2301 \
  --config zh \
  --text-column text \
  --output "$OUTPUT" \
  --append \
  --min-length 20 \
  --max-lines 1200000 \
  --hf-endpoint "$HF_ENDPOINT_VALUE" \
  --hf-timeout "$HF_TIMEOUT" \
  --hf-retries "$HF_RETRIES"

run_fetch "[4/4] 追加 fjcanyue 中文维基（可达则增量补充）..." \
  --dataset fjcanyue/wikipedia-zh-cn \
  --data-files wikipedia-zh-cn-20260201.json \
  --text-column text \
  --output "$OUTPUT" \
  --append \
  --min-length 20 \
  --max-lines 1200000 \
  --hf-endpoint "$HF_ENDPOINT_VALUE" \
  --hf-timeout "$HF_TIMEOUT" \
  --hf-retries "$HF_RETRIES"

echo "done: $OUTPUT"
echo "sources_succeeded=$SUCCEEDED"
echo "sources_failed=$FAILED"
wc -l "$OUTPUT"
