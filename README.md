# GPT-2 Pretrain Course Project

从零实现一个最小可交付的 GPT-2 风格中文预训练项目，覆盖：

- 语料清洗与切分
- WordPiece tokenizer 训练
- GPT decoder-only 模型实现
- 训练/验证损失记录
- 文本生成与解码策略对比
- 模型大小与数据规模对比实验

## 目录结构

```text
.
├── configs/
├── data/
│   ├── processed/
│   └── raw/
├── outputs/
├── reports/
├── scripts/
├── src/
│   └── gpt2_pretrain/
└── tests/
```

## 快速开始

1. 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

2. 准备原始语料

把中文语料按行写入 `data/raw/corpus.txt`。仓库内已提供一个小样例 `data/raw/sample_corpus.txt` 用于 smoke test。

3. 清洗语料

```bash
python3 scripts/prepare_corpus.py \
  --input data/raw/sample_corpus.txt \
  --output data/processed/corpus_clean.txt
```

4. 训练 tokenizer

```bash
python3 scripts/train_tokenizer.py \
  --config configs/tiny_gpt.yaml \
  --input data/processed/corpus_clean.txt
```

5. 将文本编码成训练集

```bash
python3 scripts/build_dataset.py \
  --config configs/tiny_gpt.yaml \
  --input data/processed/corpus_clean.txt
```

6. 启动训练

```bash
python3 scripts/train.py --config configs/tiny_gpt.yaml
```

7. 生成文本

```bash
python3 scripts/generate.py \
  --config configs/tiny_gpt.yaml \
  --checkpoint outputs/tiny_gpt/checkpoints/best.pt \
  --prompt "人工智能的发展" \
  --strategy top_p
```

8. 绘制损失曲线

```bash
python3 scripts/plot_losses.py --run-dir outputs/tiny_gpt
```

## 推荐配置

- `configs/tiny_gpt.yaml`: 适合 8GB 显存本机调试与作业跑通
- `configs/gpt2_mini_4090.yaml`: 适合 24GB 4090 远程服务器
- `configs/gpt2_small_a800.yaml`: 适合 80GB A800 展示更大模型

## 推荐实验

1. 解码策略对比：`greedy / top_k / top_p`
2. 模型大小对比：`tiny_gpt / gpt2_mini`
3. 数据规模对比：`small_corpus / full_corpus`

## 结果产物

- `outputs/<run_name>/train_loss.csv`
- `outputs/<run_name>/valid_loss.csv`
- `outputs/<run_name>/metrics.json`
- `outputs/<run_name>/samples/*.txt`
- `outputs/<run_name>/checkpoints/*.pt`

## 注意事项

- 当前默认使用 PyTorch 原生训练循环，不依赖 HuggingFace Trainer。
- tokenizer 实现选择了 `WordPiece + BertPreTokenizer`，对中文文本比 ByteLevel 更稳定。
- 8GB 显存建议优先使用 `tiny_gpt.yaml`，必要时降低 batch size 并提高梯度累积。
- 真实课程提交前，请使用你自己的中文语料替换样例语料。
