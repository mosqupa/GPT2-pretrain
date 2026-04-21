# 《生成式 AI》课程作业报告模板

## 1. 任务目标

- 从零实现 GPT-2 风格 decoder-only 语言模型
- 完成中文语料预处理、训练、验证和文本生成
- 分析解码策略、数据规模和模型大小对结果的影响

## 2. 数据集与预处理

- 语料来源：
- 样本数量：
- 清洗规则：
- 训练/验证划分：
- tokenizer 类型与词表大小：

## 3. 模型结构

- 模型类型：GPT-2 风格 decoder-only Transformer
- 关键参数：
- 位置编码：
- 注意力机制：
- 参数量：

## 4. 训练设置

- 硬件环境：
- batch size / grad accumulation：
- learning rate / warmup：
- optimizer：
- 训练轮数或总 step：

## 5. 实验结果

### 5.1 损失曲线

- 插入 train/valid loss 曲线图

### 5.2 解码策略对比

| 策略 | 生成样例 | 观察 |
| --- | --- | --- |
| greedy |  |  |
| top-k |  |  |
| top-p |  |  |

### 5.3 模型大小对比

| 模型 | 参数量 | valid loss | ppl | 观察 |
| --- | --- | --- | --- | --- |
| Tiny GPT |  |  |  |  |
| GPT-2 Mini |  |  |  |  |

### 5.4 数据规模对比

| 数据规模 | valid loss | ppl | 观察 |
| --- | --- | --- | --- |
| small |  |  |  |
| full |  |  |  |

## 6. 生成案例

- 提示词：
- 生成结果：
- 质量分析：

## 7. 问题分析与改进

- 训练中遇到的问题
- 生成质量的局限
- 后续可尝试的优化方向

## 8. 核心代码说明

- tokenizer 训练代码
- causal self-attention 代码
- 训练循环代码
- 文本生成代码
