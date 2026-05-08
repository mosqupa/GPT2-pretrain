# 核心脚本索引

只保留“语料获取与清洗 / 数据构建 / 预训练 / 指令微调 / 推理验证”的最小闭环。默认请使用根目录下的 `scripts/*.py` 与 `scripts/*.sh` 作为正式入口。

## 数据准备（语料）
- 获取中文语料：
  `python3 scripts/fetch_corpus_hf.py --dataset 0xDing/wikipedia-cn-20230720-filtered --text-column completion --output data/raw/corpus.txt --max-lines 800000 --min-length 20 --hf-endpoint https://hf-mirror.com --hf-timeout 120 --hf-retries 10`
- 追加本地语料：
  `python3 scripts/add_corpus.py /path/to/new_corpus.txt`
- 清洗语料：
  `python3 scripts/prepare_corpus.py --input data/raw/corpus.txt --output data/processed/gpt2_mini_4090/corpus_clean.txt`
- 训练分词器：
  `python3 scripts/train_tokenizer.py --config configs/gpt2_mini_4090.yaml --input data/processed/gpt2_mini_4090/corpus_clean.txt`
- 构建 memmap 数据集：
  `python3 scripts/build_dataset.py --config configs/gpt2_mini_4090.yaml --input data/processed/gpt2_mini_4090/corpus_clean.txt`

## 预训练（Pretrain）
- 启动训练：
  `python3 scripts/train.py --config configs/gpt2_mini_4090.yaml`
- 一键跑通 4090 配置：
  `bash scripts/run_4090_pipeline.sh`
- 绘制损失曲线：
  `python3 scripts/plot_losses.py --run-dir outputs/<run1> outputs/<run2> --output-dir outputs/<target_run>`
- 文本生成验证：
  `python3 scripts/generate.py --config configs/gpt2_mini_4090.yaml --checkpoint outputs/<run>/checkpoints/best.pt --prompt "人工智能"`

## 指令微调（SFT）
- 拉取/归一化中文指令数据：
  `python3 scripts/fetch_sft_data.py --dataset PKU-Alignment/Align-Anything-Instruction-100K-zh --output data/raw/sft_zh.jsonl --hf-endpoint https://hf-mirror.com --hf-timeout 120 --hf-retries 10`
- 合并本地 SFT 数据：
  `python3 scripts/add_sft_data.py /path/to/file.jsonl`
- 切分训练/验证集：
  `python3 scripts/prepare_sft_dataset.py --config configs/gpt2_mini_4090_sft.yaml`
- 启动 SFT 训练：
  `python3 scripts/train_sft.py --config configs/gpt2_mini_4090_sft.yaml`
- 交互对话：
  `python3 scripts/chat_sft.py`
- 离线兜底构造 SFT：
  `python3 scripts/bootstrap_sft_from_corpus.py`

## 分类目录说明
- `scripts/data/`：数据准备相关脚本的分类别名入口
- `scripts/pretrain/`：预训练相关脚本的分类别名入口
- `scripts/sft/`：SFT 相关脚本的分类别名入口

## 非核心（可忽略）
- 实验对比、演示或模板相关内容不影响核心闭环，可在课程提交路径中忽略。
