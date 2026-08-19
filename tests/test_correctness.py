"""Correctness tests: causal mask, weight tying, top-k sampling, data split."""
import torch

from config import GPT2Config
from model import GPT2
from data import DataProcessor
from tokenizer import CharTokenizer


def _cfg():
    cfg = GPT2Config()
    cfg.vocab_size = 100
    cfg.n_embd = 32
    cfg.n_head = 2
    cfg.n_layer = 1
    return cfg


def test_causal_mask():
    cfg = _cfg()
    cfg.n_head = 1
    model = GPT2(cfg)
    b = model.transformer.h[0].attn.bias  # (1, 1, T, T)
    T = cfg.block_size
    # the causal mask must be lower-triangular: 1 at/after the diagonal, 0 above
    expected = torch.tril(torch.ones(T, T))
    assert torch.equal(b[0, 0, :T, :T], expected)


def test_weight_tying():
    cfg = _cfg()
    model = GPT2(cfg)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_topk_sampling():
    cfg = _cfg()
    model = GPT2(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    gen = model.generate(idx, max_new_tokens=10, top_k=1)
    assert gen.shape[1] == 8 + 10
    # top_k=1 is greedy argmax => deterministic across identical runs
    gen2 = model.generate(idx, max_new_tokens=10, top_k=1)
    assert torch.equal(gen, gen2)


def test_data_split_integrity(tmp_path):
    text = "hello world " * 100  # 1200 chars
    tok = CharTokenizer()
    tok.train(text)
    proc = DataProcessor(tok)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    train_n, val_n = proc.process_directory(str(src), str(out), train_ratio=0.9)
    assert train_n == int(len(text) * 0.9)
    assert val_n == len(text) - int(len(text) * 0.9)
    assert (out / "train.bin").exists() and (out / "val.bin").exists()