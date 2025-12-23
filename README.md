# GPT-2 从零实现

一个参考 nanoGPT 风格的简洁 GPT-2 实现，只使用 PyTorch，包含完整的训练和推理流程。

## 项目结构

```
gpt2_from_scratch/
├── config.py          # 配置文件
├── model.py           # GPT-2 模型架构
├── tokenizer.py       # Tokenizer实现
├── data.py            # 数据加载和处理
├── train.py           # 训练脚本
├── utils.py           # 工具函数
├── generate.py        # 文本生成脚本
├── prepare_data.py    # 数据准备脚本
└── README.md          # 项目说明
```

## 特性

✅ **简洁的模型实现**：清晰的 Transformer 架构实现  
✅ **高效的数据加载**：使用 memory-mapped arrays  
✅ **完整的训练流程**：包含 warmup、cosine decay、gradient clipping  
✅ **Checkpoint 管理**：自动保存和恢复训练  
✅ **Tensorboard 支持**：可视化训练过程  
✅ **多种 Tokenizer**：支持字符级、BPE 和 GPT-2 tokenizer  
✅ **文本生成**：支持交互式和批量生成  

## 安装依赖

```bash
pip install torch numpy tensorboard
# 可选：使用 GPT-2 tokenizer
pip install tiktoken
# 或者
pip install transformers
```

## 快速开始

### 1. 准备数据

**使用莎士比亚数据集（快速测试）：**

```bash
python prepare_data.py --dataset shakespeare
```

**使用自定义数据：**

```bash
python prepare_data.py --dataset custom --input /path/to/your/text.txt --tokenizer gpt2
```

**使用 OpenWebText（大规模训练）：**

```bash
python prepare_data.py --dataset openwebtext
```

### 2. 修改配置

编辑 `config.py` 文件，调整训练参数：

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
    data_dir = './data/shakespeare'  # 修改为你的数据目录
```

### 3. 开始训练

```bash
python train.py
```

训练日志会保存在 `./runs` 目录，checkpoint 保存在 `./checkpoints` 目录。

### 4. 监控训练

使用 Tensorboard 查看训练过程：

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
    "The future of AI"
]

batch_generate(prompts, output_file='samples.txt')
```

## 模型配置

项目提供了多个预定义的 GPT-2 模型配置：

| 模型 | 层数 | 注意力头 | 嵌入维度 | 参数量 |
|------|------|----------|----------|--------|
| GPT-2 | 12 | 12 | 768 | 124M |
| GPT-2 Medium | 24 | 16 | 1024 | 350M |
| GPT-2 Large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1558M |

在 `config.py` 中使用：

```python
from config import GPT2Config, GPT2_CONFIGS

config = GPT2Config()
# 使用 GPT-2 Medium 配置
for k, v in GPT2_CONFIGS['gpt2-medium'].items():
    setattr(config, k, v)
```

## 训练技巧

### 学习率调度

项目使用 warmup + cosine decay 学习率调度：

```python
config.warmup_iters = 2000      # warmup 迭代次数
config.lr_decay_iters = 100000  # decay 结束迭代次数
config.learning_rate = 6e-4     # 最大学习率
config.min_lr = 6e-5            # 最小学习率
```

### 梯度裁剪

```python
config.grad_clip = 1.0  # 梯度裁剪阈值
```

### 混合精度训练

```python
config.dtype = torch.float16  # 使用 FP16
```

### 模型编译（PyTorch 2.0+）

```python
config.compile = True  # 使用 torch.compile 加速
```

## 数据格式

数据需要预处理为二进制格式：

1. **train.bin**：训练数据
2. **val.bin**：验证数据

每个文件包含 uint16 类型的 token IDs。

## Checkpoint 管理

训练过程中会自动保存：

- `checkpoint.pt`：最新的 checkpoint
- `best_checkpoint.pt`：验证集上最好的 checkpoint
- `checkpoint_iter_N.pt`：每 N 次迭代的 checkpoint
- `final_checkpoint.pt`：训练结束时的 checkpoint

Checkpoint 包含：
- 模型权重
- 优化器状态
- 训练迭代次数
- 最佳验证损失
- 配置参数

## 性能优化

1. **使用 CUDA**：在 GPU 上训练会快很多
2. **增加 batch size**：充分利用 GPU 内存
3. **使用 torch.compile**：PyTorch 2.0+ 可以加速训练
4. **混合精度训练**：使用 FP16 减少内存占用
5. **gradient accumulation**：模拟更大的 batch size

## 常见问题

**Q: 显存不足怎么办？**

A: 减小 `batch_size`、`block_size` 或 `n_embd`。

**Q: 训练速度慢？**

A: 确保使用 GPU，启用 `compile=True`，使用混合精度训练。

**Q: 如何继续训练？**

A: 训练会自动从 `checkpoint.pt` 恢复，无需额外操作。

**Q: 如何使用自己的数据？**

A: 使用 `prepare_data.py --dataset custom --input your_data.txt`

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy 的简洁实现

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！