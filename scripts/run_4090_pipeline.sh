#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/gpt2_mini_4090.yaml"
RAW_CORPUS="data/raw/corpus.txt"
CLEAN_CORPUS="data/processed/gpt2_mini_4090/corpus_clean.txt"

echo "[1/4] 清洗语料..."
python3 scripts/prepare_corpus.py \
  --input "$RAW_CORPUS" \
  --output "$CLEAN_CORPUS"

# echo "[2/4] 训练 tokenizer..."
# python3 scripts/train_tokenizer.py \
#   --config "$CONFIG" \
#   --input "$CLEAN_CORPUS"

echo "[3/4] 构建训练数据..."
python3 scripts/build_dataset.py \
  --config "$CONFIG" \
  --input "$CLEAN_CORPUS"

echo "[4/4] 启动训练..."
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/gpt2_mini_4090.yaml
