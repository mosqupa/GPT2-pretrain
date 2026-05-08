#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/gpt2_mini_4090_sft.yaml}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-outputs/gpt2_mini_4090_20260429_054045/checkpoints/last.pt}"
TRAIN_CONFIG="$CONFIG"

if [[ -n "$PRETRAIN_CKPT" ]]; then
  echo "[info] 使用预训练 checkpoint: $PRETRAIN_CKPT"
  TMP_CONFIG="$(mktemp /tmp/gpt2_sft_XXXX.yaml)"
  python3 - <<PY
from pathlib import Path
import yaml

config_path = Path("$CONFIG")
tmp_path = Path("$TMP_CONFIG")
payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
payload["train"]["resume_from"] = "$PRETRAIN_CKPT"
tmp_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY
  TRAIN_CONFIG="$TMP_CONFIG"
fi

echo "[1/3] 抓取中文 SFT 数据..."
if python3 scripts/fetch_sft_data.py \
  --output data/raw/sft_zh.jsonl \
  --max-samples 50000 \
  --hf-endpoint https://hf-mirror.com \
  --hf-timeout 120 \
  --hf-retries 10; then
  echo "[info] 已从外部数据源抓取 SFT 数据。"
else
  echo "[warn] 外网抓取失败，回退到本地语料构造离线 SFT 数据..."
  python3 scripts/bootstrap_sft_from_corpus.py \
    --input data/raw/corpus.txt \
    --output data/raw/sft_zh.jsonl \
    --max-samples 50000
fi

echo "[2/3] 切分 train/valid..."
python3 scripts/prepare_sft_dataset.py --config "$TRAIN_CONFIG"

echo "[3/3] 启动 SFT..."
python3 scripts/train_sft.py --config "$TRAIN_CONFIG"
