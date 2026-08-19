"""Component / smoke tests: config, model, tokenizers, LR schedule, checkpoints."""
import pytest
import torch

from config import GPT2_CONFIGS, GPT2Config
from model import GPT2
from tokenizer import CharTokenizer, SimpleBPETokenizer, get_tokenizer
from utils import get_lr, load_checkpoint, save_checkpoint


def make_tiny_config():
    cfg = GPT2Config()
    cfg.vocab_size = 100
    cfg.block_size = 32
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 32
    return cfg


def test_config_presets():
    assert GPT2_CONFIGS['gpt2'] == {'n_layer': 12, 'n_head': 12, 'n_embd': 768}


def test_config_dict_roundtrip():
    """Class-attribute defaults must survive to_dict/from_dict (checkpoint config)."""
    cfg = GPT2Config()
    d = cfg.to_dict()
    assert d['vocab_size'] == 50257  # a class-level default is captured
    cfg.n_layer = 3
    restored = GPT2Config.from_dict(cfg.to_dict())
    assert restored.n_layer == 3
    assert restored.block_size == 1024


def test_forward_pass_shape():
    cfg = make_tiny_config()
    model = GPT2(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.shape == ()


def test_backward_runs():
    cfg = make_tiny_config()
    model = GPT2(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer()
    tok.train("hello world")
    ids = tok.encode("hello")
    assert ids
    assert tok.decode(ids) == "hello"


def test_simple_bpe_roundtrip():
    tok = SimpleBPETokenizer(vocab_size=280)
    tok.train("the quick brown fox jumps over the lazy dog", num_merges=20)
    ids = tok.encode("the quick")
    assert tok.decode(ids) == "the quick"


def test_gpt2_tokenizer_optional():
    try:
        tok = get_tokenizer('gpt2')
    except ImportError:
        pytest.skip("tiktoken/transformers not installed")
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"


def test_lr_schedule():
    cfg = GPT2Config()
    cfg.warmup_iters = 100
    cfg.lr_decay_iters = 1000
    cfg.learning_rate = 1.0
    cfg.min_lr = 0.1
    assert get_lr(0, cfg) == pytest.approx(0.0)
    assert get_lr(50, cfg) == pytest.approx(0.5)
    assert get_lr(100, cfg) == pytest.approx(1.0)
    assert get_lr(550, cfg) == pytest.approx(0.55)
    assert get_lr(1001, cfg) == pytest.approx(0.1)


def test_checkpoint_roundtrip(tmp_path):
    cfg = make_tiny_config()
    cfg.checkpoint_dir = str(tmp_path)
    model = GPT2(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(model, optimizer, 1, 1.234, cfg)
    model2 = GPT2(cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    it, blv = load_checkpoint(model2, optimizer2, cfg)
    assert it == 1
    assert blv == pytest.approx(1.234)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)