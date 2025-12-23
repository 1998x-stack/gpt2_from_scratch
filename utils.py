"""
Utility Functions
包含checkpoint保存/加载、学习率调度等工具函数
"""
import os
import torch
import json
from contextlib import nullcontext


def get_lr(it, config):
    """
    学习率调度：warmup + cosine decay
    参考nanoGPT的实现
    """
    # 1) Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * torch.pi)))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def save_checkpoint(model, optimizer, iter_num, best_val_loss, config, filename='checkpoint.pt'):
    """保存checkpoint"""
    checkpoint_dir = config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config.to_dict(),
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, config, filename='checkpoint.pt'):
    """加载checkpoint"""
    checkpoint_path = os.path.join(config.checkpoint_dir, filename)
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
    return iter_num, best_val_loss


def save_model_only(model, config, filename='model.pt'):
    """只保存模型权重（不包括optimizer）"""
    checkpoint_dir = config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoint_dir, filename)
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num):
    """格式化大数字（如参数量）"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


class AverageMeter:
    """计算和存储平均值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_batch_info(tokens, block_size):
    """获取batch的统计信息"""
    info = {
        'num_tokens': len(tokens),
        'num_batches': len(tokens) // block_size,
        'size_mb': len(tokens) * 2 / 1024 / 1024,  # uint16 = 2 bytes
    }
    return info


def setup_logging(config):
    """设置日志目录"""
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)


def get_device_context(device_type, dtype):
    """
    获取设备上下文管理器
    用于混合精度训练
    """
    if device_type == 'cuda':
        return torch.amp.autocast(device_type=device_type, dtype=dtype)
    else:
        return nullcontext()


def print_training_info(config, model, train_tokens, val_tokens):
    """打印训练信息"""
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(f"Model: GPT-2")
    print(f"Parameters: {format_number(model.get_num_params())}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Block size: {config.block_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max iterations: {config.max_iters}")
    print(f"Device: {config.device}")
    print(f"Train tokens: {format_number(train_tokens)}")
    print(f"Val tokens: {format_number(val_tokens)}")
    print("=" * 80)


def save_training_config(config, filename='config.json'):
    """保存训练配置为JSON"""
    config_path = os.path.join(config.checkpoint_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to {config_path}")


def load_training_config(config, filename='config.json'):
    """从JSON加载训练配置"""
    config_path = os.path.join(config.checkpoint_dir, filename)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return type(config).from_dict(config_dict)
    return None


class GradScaler:
    """
    简单的梯度缩放器
    用于混合精度训练
    """
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._scale = 2.0 ** 16
        
    def scale(self, loss):
        if self.enabled:
            return loss * self._scale
        return loss
    
    def step(self, optimizer):
        if self.enabled:
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.grad.data.div_(self._scale)
        optimizer.step()
    
    def update(self):
        pass  # 简化版本，不动态调整scale


def clip_gradients(model, max_norm):
    """梯度裁剪"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)