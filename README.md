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

双卡训练可直接使用 `torchrun`，例如两张 4090：

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/gpt2_mini_4090.yaml
```

如果要从旧的单卡 checkpoint 继续训练，可在配置里设置 `train.resume_from` 指向具体的 `last.pt` 路径（例如 `outputs/gpt2_mini_4090_20260421_210000/checkpoints/last.pt`），再用同样的 `torchrun` 命令启动。

### 目录结构逻辑

每次执行训练脚本，系统都会在 `outputs` 目录下创建一个带时间戳的新文件夹，例如 `outputs/gpt2_mini_4090_20260421_213015/`。
- 所有的训练日志、生成样本和 Checkpoint 都会保存在该时间戳文件夹内。
- `metrics.json` 中包含 `global_step`（总步数）和 `session_steps`（本次训练步数），方便区分训练进度。
- 这种方式可以让你清晰地管理每一次实验（Session），并能通过修改配置文件中的 `resume_from` 自由选择从哪个历史版本恢复。

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
- `configs/gpt2_mini_4090.yaml`: 默认按双卡 4090 继续训练，`resume_from` 指向 `outputs/gpt2_mini_4090/checkpoints/last.pt`
- `configs/gpt2_small_a800.yaml`: 适合 80GB A800 展示更大模型
- `scripts/run_4090_pipeline.sh`: 若检测到已有 checkpoint，会跳过预处理并直接双卡续训

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
- 所有训练配置都提供了 `train.resume_from` 字段；不恢复时可设为 `null`，恢复时改成目标 checkpoint 路径。
- 真实课程提交前，请使用你自己的中文语料替换样例语料。
