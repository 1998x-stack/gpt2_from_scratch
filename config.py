"""GPT-2 configuration.

Holds model architecture, training, and system settings, plus a registry of
predefined GPT-2 model sizes. 配置文件：模型结构 / 训练超参 / 数据集路径等。
"""
import torch


def _is_data_attr(v) -> bool:
    """True if ``v`` is a config field value (not a method/classmethod)."""
    from types import FunctionType

    return not callable(v) and not isinstance(v, (FunctionType, classmethod, staticmethod))


class GPT2Config:
    # Model architecture
    vocab_size = 50257  # GPT-2 vocab size
    block_size = 1024   # context length
    n_layer = 12        # number of transformer blocks
    n_head = 12         # number of attention heads
    n_embd = 768        # embedding dimension
    dropout = 0.1       # dropout probability
    bias = True         # use bias in Linear and LayerNorm
    
    # Training hyperparameters
    batch_size = 12     # batch size
    learning_rate = 6e-4
    max_iters = 100000  # total training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0     # gradient clipping
    
    # Learning rate decay
    decay_lr = True
    warmup_iters = 2000
    lr_decay_iters = 100000
    min_lr = 6e-5
    
    # Evaluation
    eval_interval = 500
    eval_iters = 200
    log_interval = 10
    
    # Checkpoint
    save_interval = 1000
    checkpoint_dir = './checkpoints'
    
    # System
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    compile = False  # use torch.compile (PyTorch 2.0+)
    
    # Data
    data_dir = './data'
    dataset = 'openwebtext'

    # Tensorboard
    log_dir = './runs'

    def __init__(self) -> None:
        """Snapshot class-level defaults onto the instance.

        This makes :meth:`to_dict` / :meth:`from_dict` round-trip correctly
        (they read ``instance.__dict__``), so a checkpoint's config is saved
        even when the model is driven purely by class attributes.
        """
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_') and _is_data_attr(v):
                setattr(self, k, v)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GPT2Config":
        """Create a config from a dict, overriding default attributes."""
        config = cls()
        for k, v in config_dict.items():
            setattr(config, k, v)
        return config

    def to_dict(self) -> dict:
        """Return the config's attributes as a dict (excluding private keys)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# GPT-2 模型变体配置
GPT2_CONFIGS = {
    'gpt2': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},        # 124M params
    'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},  # 350M params
    'gpt2-large': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},   # 774M params
    'gpt2-xl': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600},      # 1558M params
}