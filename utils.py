"""Utility functions: checkpointing, LR scheduling, logging, seeding, device context.

Docstrings use concise English with a Chinese usage note where it aids the
learning goal. 本模块是训练流程的工具层（checkpoint / 学习率 / 日志等）。
"""
from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch


def get_lr(it: int, config) -> float:
    """Return the learning rate at iteration ``it`` (linear warmup + cosine decay).

    参考 nanoGPT 的学习率调度：线性 warmup -> cosine decay -> min_lr。
    """
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (
        config.lr_decay_iters - config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * torch.pi)))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def save_checkpoint(model, optimizer, iter_num: int, best_val_loss: float, config, filename: str = 'checkpoint.pt') -> None:
    """Save a training checkpoint (model weights + optimizer state + config)."""
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


def load_checkpoint(model, optimizer, config, filename: str = 'checkpoint.pt') -> tuple[int, float]:
    """Load a checkpoint into ``model`` and ``optimizer``.

    Returns:
        (iter_num, best_val_loss) restored from the checkpoint. If no
        checkpoint exists, returns ``(0, float('inf'))``.
    """
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


def save_model_only(model, config, filename: str = 'model.pt') -> None:
    """Save only model weights (no optimizer state) to ``checkpoint_dir``."""
    checkpoint_dir = config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(checkpoint_dir, filename)
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)


def count_parameters(model) -> int:
    """Return the total number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """Format a large integer with K/M/B suffixes (e.g. 1.23M)."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


class AverageMeter:
    """Track a running average of observed values."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def setup_logging(config) -> None:
    """Create the log and checkpoint directories if missing."""
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)


def get_device_context(device_type: str, dtype) -> Any:
    """Return a context manager suitable for mixed-precision training.

    On CUDA returns ``torch.amp.autocast``; on CPU returns a null context.
    """
    if device_type == 'cuda':
        return torch.amp.autocast(device_type=device_type, dtype=dtype)
    return nullcontext()


def print_training_info(config, model, train_tokens: int, val_tokens: int) -> None:
    """Print a formatted summary of the training configuration."""
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


def save_training_config(config, filename: str = 'config.json') -> None:
    """Dump the training config to JSON in ``checkpoint_dir``."""
    config_path = os.path.join(config.checkpoint_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to {config_path}")


def load_training_config(config, filename: str = 'config.json') -> Any:
    """Load a config from JSON in ``checkpoint_dir``; returns None if absent."""
    config_path = os.path.join(config.checkpoint_dir, filename)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return type(config).from_dict(config_dict)
    return None


def clip_gradients(model, max_norm: float) -> None:
    """Clip all gradients to a maximum L2 norm."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def set_seed(seed: int) -> None:
    """Seed PyTorch, NumPy and Python's random for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)