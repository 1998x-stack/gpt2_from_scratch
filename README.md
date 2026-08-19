# GPT-2 从零实现

一个参考 [nanoGPT](https://github.com/karpathy/nanoGPT) 风格的简洁 GPT-2 实现，仅使用 PyTorch，包含完整的**训练、生成与数据准备**流程。适合学习 Transformer 语言模型的内部结构与训练技巧。

> A clean, from-scratch GPT-2 in PyTorch (nanoGPT-style) with full training / generation / data pipelines. Great for learning the internals of a decoder-only Transformer.

## 特性 / Features

✅ **简洁的模型实现**：清晰的 Transformer 架构（`LayerNorm` / `CausalSelfAttention` / `MLP` / `Block`），保留 nanoGPT 的命名便于对照原文  
✅ **高效的数据加载**：使用 memory-mapped arrays（`np.memmap`），可处理超大数据集  
✅ **完整的训练流程**：warmup + cosine decay、gradient clipping、混合精度（CUDA）  
✅ **训练优化**：支持 `torch.compile` 编译加速，可搭配 gradient accumulation（见性能优化）  
✅ **Checkpoint 管理**：自动保存与恢复训练  
✅ **Tensorboard 支持**：可视化损失与学习率  
✅ **多种 Tokenizer**：字符级（char）、字节级 BPE、官方 GPT-2 tokenizer  
✅ **文本生成**：温度采样、top-k 采样、交互式 / 批量生成  
✅ **测试套件**：`pytest` 覆盖组件、功能正确性与端到端训练

## 安装依赖

```bash
pip install torch numpy tensorboard
# 可选：使用官方 GPT-2 tokenizer
pip install tiktoken
# 或
pip install transformers

# 开发 / 测试依赖（pytest、ruff）
pip install -r requirements-dev.txt
```

## 项目结构

```
gpt2_from_scratch/
├── config.py          # 配置文件（模型结构 / 训练超参 / 预设模型尺寸）
├── model.py           # GPT-2 模型架构
├── tokenizer.py       # 管线用 Tokenizer（Char / BPE / GPT-2）
├── simple_tokenizer.py # 字节级 BPE 教学演示（仅供学习，不参与管线）
├── data.py            # 数据加载（memmap）与预处理
├── train.py           # 训练脚本
├── utils.py           # 工具函数（checkpoint、LR 调度、日志、种子）
├── generate.py        # 文本生成（单次 / 交互 / 批量）
├── prepare_data.py    # 数据准备脚本
├── quickstart.py      # 快速开始示例（mini 训练 + 组件测试）
├── tests/             # pytest 测试套件
│   ├── test_components.py   # 组件 / 冒烟测试
│   ├── test_correctness.py  # 因果掩码、权重共享、top-k、数据划分
│   └── test_end_to_end.py   # 端到端 mini 训练测试
├── requirements.txt    # 运行时依赖
├── requirements-dev.txt# 开发依赖
└── pytest.ini          # pytest 配置
```

## 快速开始

### 1. 准备数据

**莎士比亚数据集（快速测试）：**

```bash
python prepare_data.py --dataset shakespeare
```

**自定义数据（单个文件或目录）：**

```bash
python prepare_data.py --dataset custom --input /path/to/your/text.txt --tokenizer gpt2
```

**OpenWebText（大规模训练）：**

```bash
python prepare_data.py --dataset openwebtext
```

> `--tokenizer` 可选 `char` / `bpe` / `gpt2`（默认 `gpt2`）。

### 2. 修改配置

编辑 `config.py` 调整训练参数：

```python
class GPT2Config:
    # 模型架构
    vocab_size = 50257
    block_size = 1024  # 上下文长度
    n_layer = 12       # Transformer 层数
    n_head = 12        # 注意力头数
    n_embd = 768       # 嵌入维度

    # 训练超参数
    batch_size = 12
    learning_rate = 6e-4
    max_iters = 100000

    # 数据路径
    data_dir = './data/shakespeare'  # 改为你的数据目录
```

### 3. 开始训练

```bash
python train.py
```

训练日志保存在 `./runs`，checkpoint 保存在 `./checkpoints`。

### 4. 监控训练

```bash
tensorboard --logdir=./runs
```

### 5. 生成文本

**单次生成：**

```bash
python generate.py --prompt "Once upon a time" --max_new_tokens 200
```

**交互式生成：**

```bash
python generate.py --interactive
```

**批量生成：**

```python
from generate import batch_generate

prompts = [
    "Once upon a time",
    "In a galaxy far, far away",
    "The future of AI",
]
batch_generate(prompts, output_file='samples.txt')
```

## 模型配置

| 模型 | 层数 | 注意力头 | 嵌入维度 | 参数量 |
|------|------|----------|----------|--------|
| GPT-2 | 12 | 12 | 768 | 124M |
| GPT-2 Medium | 24 | 16 | 1024 | 350M |
| GPT-2 Large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1558M |

在代码中使用预设尺寸：

```python
from config import GPT2Config, GPT2_CONFIGS

config = GPT2Config()
for k, v in GPT2_CONFIGS['gpt2-medium'].items():
    setattr(config, k, v)
```

## 训练技巧

**学习率调度**（warmup + cosine decay）：

```python
config.warmup_iters = 2000      # warmup 迭代次数
config.lr_decay_iters = 100000  # decay 结束迭代次数
config.learning_rate = 6e-4     # 最大学习率
config.min_lr = 6e-5            # 最小学习率
```

**梯度裁剪：**

```python
config.grad_clip = 1.0  # 梯度裁剪阈值（0 表示关闭）
```

**混合精度训练：**

```python
config.dtype = torch.float16  # 使用 FP16（仅 CUDA 生效，CPU 自动回退）
```

**模型编译（PyTorch 2.0+）：**

```python
config.compile = True  # 使用 torch.compile 加速
```

## 数据格式

数据被预处理为二进制格式：

1. `train.bin`：训练数据
2. `val.bin`：验证数据

每个文件包含 `uint16` 类型的 token IDs（每 token 2 字节）。`DataLoaderWrapper` 通过 `np.memmap` 按需读取，避免加载整个数组到内存。

## Checkpoint 管理

训练过程中自动保存：

- `checkpoint.pt`：最新 checkpoint（用于续训）
- `best_checkpoint.pt`：验证集上最好的 checkpoint（用于生成）
- `checkpoint_iter_N.pt`：每 N 次迭代的 checkpoint
- `final_checkpoint.pt`：训练结束时的 checkpoint

Checkpoint 包含：模型权重、优化器状态、训练迭代次数、最佳验证损失、配置参数。

## 测试

运行全部测试（CPU，约 1-2 秒）：

```bash
python -m pytest -v
```

- `tests/test_components.py`：模型前向/反向、tokenizer 编解码、LR 调度、checkpoint 存取
- `tests/test_correctness.py`：因果掩码、权重共享、top-k 采样、数据划分
- `tests/test_end_to_end.py`：在小型语料上训练一个迷你 GPT-2 并验证生成

## 性能优化

1. **使用 CUDA**：GPU 上训练效率远高于 CPU
2. **增大 batch size**：更充分利用 GPU 显存
3. **使用 `torch.compile`**：PyTorch 2.0+ 可显著加速
4. **混合精度训练**：FP16 减少显存占用与计算量
5. **gradient accumulation**：用更小的显存模拟更大的 batch size

## 常见问题

**Q: 显存不足怎么办？**
A: 减小 `batch_size`、`block_size` 或 `n_embd`。

**Q: 训练速度慢？**
A: 确保使用 GPU，开启 `compile=True`，并启用混合精度训练。

**Q: 如何继续训练？**
A: 自动从 `checkpoints/checkpoint.pt` 恢复，无需额外操作。

**Q: 如何使用自己的数据？**
A: 使用 `prepare_data.py --dataset custom --input your_data.txt`。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原论文
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 论文
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy 的简洁实现

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！