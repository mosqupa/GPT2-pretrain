# scripts 目录说明

`scripts/` 现在采用两层结构：

## 1. 根目录脚本：主入口
- 日常运行默认使用 `scripts/*.py` 与 `scripts/*.sh`
- 这些文件是当前仓库的正式入口，README 中的命令也都指向这里

## 2. 子目录脚本：分类别名入口
- `scripts/data/`
- `scripts/pretrain/`
- `scripts/sft/`
- 这些子目录用于按功能分类查看脚本
- 其中脚本会尽量保持与根目录同名入口一致，便于检索和后续继续整理

## 预训练主链路
- `fetch_corpus_hf.py`：从 Hugging Face 拉取中文原始语料
- `add_corpus.py`：将本地新增语料追加到 `data/raw/corpus.txt`
- `prepare_corpus.py`：清洗原始语料
- `train_tokenizer.py`：训练 WordPiece tokenizer
- `build_dataset.py`：将清洗后的语料编码为 `train.bin / valid.bin`
- `train.py`：启动预训练
- `generate.py`：用预训练模型做文本生成
- `plot_losses.py`：绘制 train / valid loss 曲线，可合并多段续训记录

## 指令微调主链路
- `fetch_sft_data.py`：拉取并归一化中文 SFT 数据
- `add_sft_data.py`：将手动上传的 SFT 数据合并进总数据集
- `prepare_sft_dataset.py`：切分 SFT 训练集与验证集
- `train_sft.py`：启动指令微调训练
- `chat_sft.py`：交互式对话脚本，可选择不同 checkpoint

## 数据补救/离线兜底
- `bootstrap_sft_from_corpus.py`：在外部 SFT 数据拿不到时，用现有中文语料构造一份离线 SFT 数据

## 一键运行脚本
- `run_4090_pipeline.sh`：一键执行预训练主流程
- `run_4090_sft.sh`：一键执行 SFT 主流程

## 运行依赖
- `_bootstrap.py`：负责把 `src/` 加入 Python 路径，供根目录脚本直接运行

## 建议使用顺序

### 预训练
1. `fetch_corpus_hf.py`
2. `prepare_corpus.py`
3. `train_tokenizer.py`
4. `build_dataset.py`
5. `train.py`
6. `generate.py` / `plot_losses.py`

### 指令微调
1. `fetch_sft_data.py` 或 `bootstrap_sft_from_corpus.py`
2. `prepare_sft_dataset.py`
3. `train_sft.py`
4. `chat_sft.py`
