# GPT-2 from scratch — Professional Code & Docs Enhancement (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix bugs, clean up, add type hints + Google-style docstrings, add a pytest suite, packaging files, and a polished bilingual README — while preserving the nanoGPT from-scratch learning identity.

**Architecture:** Flat-module layout is preserved. Each file keeps its single responsibility (model / config / data / utils / train / generate / prepare / tokenizer). Cleanup is done in-place; a new `tests/` directory holds the suite; packaging files (`requirements*.txt`, `.gitignore`, `pytest.ini`) are added at the root.

**Tech Stack:** Python 3.9+, PyTorch, NumPy, pytest, ruff (dev). Optional: tiktoken/transformers.

## Global Constraints

- **Bilingual:** code comments/docstrings in concise English; README Chinese-primary with English notes. Do not translate existing Chinese pedagogic comments inside `simple_tokenizer.py` back to English — keep its teaching style, just fix bugs/docstring.
- **No renaming** of core classes/methods (`GPT2`, `Block`, `CausalSelfAttention`, `MLP`, `LayerNorm`, `c_attn`, `c_proj`). Flat module layout preserved (no `src/`, no `pyproject.toml`, no CI).
- **No behavior change** to the training pipeline semantics; only bug fixes and cleanup.
- Runtime deps: `torch`, `numpy`, `tensorboard`. Optional: `tiktoken` / `transformers` (imported lazily).
- Tests must run on CPU (no cluster/GPU required).
- Every checkpoint/config call site must keep working signatures.

---

### Task 1: Repo hygiene files

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: lint/test tooling config consumed by all later tasks (`pytest` discovery via `pytest.ini`).

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Byte-compiled / caches
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/

# Virtual envs
.venv/
venv/

# Worktrees
.worktrees/

# Training artifacts
data/
raw_data/
temp_data/
checkpoints/
checkpoints_mini/
runs/

# IDE
.idea/
.vscode/
.DS_Store

# Misc
*.pkl
*.pt
*.log
```

- [ ] **Step 2: Write `requirements.txt`**

```text
torch>=2.0
numpy
tensorboard

# Optional tokenizers (install at least one for GPT-2 tokenizer support)
# tiktoken
# transformers
```

- [ ] **Step 3: Write `requirements-dev.txt`**

```text
-r requirements.txt
pytest>=7.0
ruff
tiktoken
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    slow: longer-running tests (e.g. end-to-end training)
```

- [ ] **Step 5: Verify**

```bash
cd <repo root> && ls -la .gitignore requirements.txt requirements-dev.txt pytest.ini
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt requirements-dev.txt pytest.ini
git commit -m "chore: add project hygiene files (gitignore, requirements, pytest config)"
```

---

### Task 2: Clean up and type-hint `utils.py`

**Files:**
- Modify: `utils.py` (rewrite; remove `GradScaler`, `get_batch_info`)

**Interfaces:**
- Consumes: `GPT2Config` from `config.py`, `torch`.
- Produces (stable signatures):
  - `get_lr(it: int, config) -> float`
  - `save_checkpoint(model, optimizer, iter_num: int, best_val_loss: float, config, filename: str = 'checkpoint.pt') -> None`
  - `load_checkpoint(model, optimizer, config, filename: str = 'checkpoint.pt') -> tuple[int, float]`
  - `save_model_only(model, config, filename: str = 'model.pt') -> None`
  - `count_parameters(model) -> int`
  - `format_number(num: int) -> str`
  - `AverageMeter` (kept as generic util)
  - `setup_logging(config) -> None`
  - `get_device_context(device_type: str, dtype) -> Any` (context manager)
  - `print_training_info(config, model, train_tokens: int, val_tokens: int) -> None`
  - `save_training_config(config, filename: str = 'config.json') -> None`
  - `load_training_config(config, filename: str = 'config.json') -> Optional[type(config)]`
  - `clip_gradients(model, max_norm: float) -> None`
  - `set_seed(seed: int) -> None`

- [ ] **Step 1: Verify current behavior before change**

```bash
cd /repo && python -c "import utils; print('ok')"
```

Measures the baseline import works. (There is no test for this module yet; behavior is verified via later test tasks and the end-to-end run.)

- [ ] **Step 2: Rewrite `utils.py`**

Remove `GradScaler` and `get_batch_info`. Keep all other functions with identical logic, but add type hints and Google-style docstrings. Example of the new header and one function:

```python
"""Utility functions: checkpointing, LR scheduling, logging, seeding.

English docstrings with a concise Chinese usage note at module level.
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
    """Return the learning rate at iteration ``it`` (warmup + cosine decay).

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
```

Keep `save_checkpoint`, `load_checkpoint`, `save_model_only`, `count_parameters`, `format_number`, `AverageMeter`, `setup_logging`, `get_device_context`, `print_training_info`, `save_training_config`, `load_training_config`, `clip_gradients`, `set_seed` with the same bodies as the current file, adding type hints and docstrings. `set_seed` already imports numpy/random inside; keep that, add seed param type:

```python
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
```

- [ ] **Step 3: Verify it still imports and runs**

```bash
cd /repo && python -c "import utils; from utils import AverageMeter; m=AverageMeter(); m.update(1.0); print(m.avg)"
```

Expected: prints `1.0`.

- [ ] **Step 4: Commit**

```bash
git add utils.py
git commit -m "refactor: clean up utils.py, drop GradScaler & dead code, add type hints"
```

---

### Task 3: Clean up and type-hint `data.py`

**Files:**
- Modify: `data.py`

**Interfaces:**
- Consumes: nothing (standalone).
- Produces (stable):
  - `class DataLoaderWrapper.__init__(self, data_dir: str, split: str, block_size: int, batch_size: int, device: str = 'cpu')`
  - `DataLoaderWrapper.get_batch(self) -> tuple[torch.Tensor, torch.Tensor]`
  - `class DataProcessor(tokenizer)`, methods `process_file(input_path, output_dir, split='train') -> int`, `process_directory(input_dir, output_dir, train_ratio=0.9) -> tuple[int, int]`
  - `download_openwebtext(data_dir)`
  - `prepare_shakespeare_data(data_dir) -> tuple[str, str]`
  - `create_dataloader(data_dir, split, block_size, batch_size, device='cpu') -> DataLoaderWrapper`
  - `estimate_loss(model, train_loader, val_loader, eval_iters: int, device) -> dict[str, float]`

- [ ] **Step 1: Remove dead code and unused imports**

Remove the entire `TextDataset` class and the `dataset`/`DataLoader` import. Remove unused `import pickle`. New header:

```python
"""Data loading and light preprocessing.

Encodes text (via an injected tokenizer) into ``uint16`` ``.bin`` files
and provides a nanoGPT-style memory-mapped :class:`DataLoaderWrapper`.
加载数据：二进制 memmap 读取, 训练时每次随机采样一个 batch。
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
```

- [ ] **Step 2: Type-hint the functions**

Add type hints to `DataLoaderWrapper.__init__`, `get_batch` (return `Tuple[torch.Tensor, torch.Tensor]`), `DataProcessor.process_file`, `process_directory`, `prepare_shakespeare_data` (return `Tuple[str, str]`), `estimate_loss` (return `dict`), and `create_dataloader`. Keep all bodies identical.

- [ ] **Step 3: Verify import**

```bash
cd /repo && python -c "import data; print('ok')"
```

- [ ] **Step 4: Commit**

```bash
git add data.py
git commit -m "refactor: tidy data.py, drop unused TextDataset/imports, add type hints"
```

---

### Task 4: Type-hint and tidy `model.py`

**Files:**
- Modify: `model.py` (docstrings + type hints only, no logic change)

**Interfaces:**
- Consumes: `GPT2Config`.
- Produces: `GPT2`, `Block`, `CausalSelfAttention`, `MLP`, `LayerNorm` (unchanged behavior). `GPT2.generate(idx, max_new_tokens, temperature=1.0, top_k=None)`. `GPT2.get_num_params(non_embedding=True) -> int`. `GPT2.configure_optimizers(weight_decay, learning_rate, betas, device_type)`.

- [ ] **Step 1: Annotate signatures & docstrings (no logic change)**

Add type hints to `LayerNorm.__init__(self, ndim: int, bias: bool)`, `forward`. `CausalSelfAttention.__init__(self, config)` and `forward(self, x)`. `MLP`, `Block`. `GPT2.__init__(self, config)`, `forward(self, idx, targets=None)` (return `tuple[torch.Tensor, torch.Tensor | None]`), `get_num_params(self, non_embedding=True) -> int`, `generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None)`, `configure_optimizers(...)`. Keep all logic byte-for-byte identical. Convert one-line mixed comments to clean English; keep the `wte`/`lm_head` weight-tying logic.

- [ ] **Step 2: Verify import + quick forward**

```bash
cd /repo && python - <<'PY'
from config import GPT2Config
from model import GPT2
c = GPT2Config(); c.vocab_size=100; c.n_layer=2; c.n_head=2; c.n_embd=32
m = GPT2(c)
import torch
x = torch.randint(0, 100, (1, 8))
logits, loss = m(x)
print(logits.shape, m.get_num_params())
PY
```

- [ ] **Step 3: Commit**

```bash
git add model.py
git commit -m "docs: type hints and docstrings for model.py (no behavior change)"
```

---

### Task 5: Type-hint and tidy `config.py`

**Files:**
- Modify: `config.py`

**Interfaces:**
- Consumes: `torch`.
- Produces: `GPT2Config` (class attributes unchanged), `GPT2Config.from_dict(config_dict)` classmethod, `GPT2Config.to_dict()` method, module-level `GPT2_CONFIGS` dict.

- [ ] **Step 1: Add `from_dict`/`to_dict` type hints and module docstring**

Add a clean English module docstring with a concise Chinese note. Annotate `from_dict(cls, config_dict: dict) -> "GPT2Config"` and `to_dict(self) -> dict`. Keep all class attribute values identical.

- [ ] **Step 2: Verify import + preset dims**

```bash
cd /repo && python - <<'PY'
from config import GPT2Config, GPT2_CONFIGS
print(GPT2_CONFIGS['gpt2-medium'])
PY
```

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "docs: type-hint and document config.py"
```

---

### Task 6: Clean and type-hint `tokenizer.py`

**Files:**
- Modify: `tokenizer.py`

**Interfaces:**
- Consumes: nothing.
- Produces (stable): `CharTokenizer`, `SimpleBPETokenizer`, `GPT2Tokenizer`, `get_tokenizer(tokenizer_type='gpt2')`.

- [ ] **Step 1: Fix `SimpleBPETokenizer`**

The class builds an `self.encoder` dict during `train()` that is never used (the real encode path uses `self.bpe_ranks`). Remove `self.encoder` / `self.decoder` redundancy: keep `self.decoder` (id->bytes) and `self.bpe_ranks`, and drop the unused `self.encoder`. Guard `encode()` so an untrained tokenizer (empty `bpe_ranks`) degrades gracefully by returning the raw byte tokens. Remove `import pickle` if now unused by this module (it is used by save/load — keep it).

Target `encode` body:

```python
def encode(self, text: str) -> list[int]:
    """Encode text into token ids via the trained BPE merges."""
    tokens = list(text.encode('utf-8'))
    while len(tokens) >= 2:
        stats = self.get_stats(tokens)
        pair = min(stats, key=lambda p: self.bpe_ranks.get(p, float('inf')))
        if pair not in self.bpe_ranks:
            break
        idx = self.bpe_ranks[pair]
        tokens = self.merge(tokens, pair, idx)
    return tokens
```

Keep `save`/`load` but drop `encoder` from the serialized dict (and handle the legacy key gracefully on load). Add type hints (`: int`, `-> ...`) and docstrings across the file.

- [ ] **Step 2: Verify round-trip**

```bash
cd /repo && python - <<'PY'
from tokenizer import SimpleBPETokenizer
t = SimpleBPETokenizer(vocab_size=280)
t.train("the quick brown fox jumps over the lazy dog", num_merges=20)
ids = t.encode("the quick")
print(t.decode(ids))
PY
```

- [ ] **Step 3: Commit**

```bash
git add tokenizer.py
git commit -m "refactor: clean SimpleBPETokenizer (drop unused encoder, guard untrained encode)"
```

---

### Task 7: Document `simple_tokenizer.py` as the teaching demo

**Files:**
- Modify: `simple_tokenizer.py` (docstring only)

**Interfaces:**
- Consumes: nothing (standalone script; not imported by the pipeline).
- Produces: nothing new.

- [ ] **Step 1: Add clarity note to module docstring**

Prepend a short note clarifying role and distinguishing from the pipeline tokenizer:

```
注意：本文件是教学演示（educational demo），不参与 train/generate 流程。
管线使用 tokenizer.py 中的 CharTokenizer / SimpleBPETokenizer / GPT2Tokenizer。
```

Keep the rest of the file (the heavy teaching commentary) intact. No logic changes.

- [ ] **Step 2: Verify it still runs standalone**

```bash
cd /repo && python simple_tokenizer.py >/dev/null && echo ok
```

Exit code 0 expected.

- [ ] **Step 3: Commit**

```bash
git add simple_tokenizer.py
git commit -m "docs: clarify simple_tokenizer.py is an educational demo, not part of the pipeline"
```

---

### Task 8: Tidy `train.py`, `generate.py`, `prepare_data.py`, `quickstart.py`

**Files:**
- Modify: `train.py`, `generate.py`, `prepare_data.py`, `quickstart.py`

**Interfaces:**
- Consumes: `GPT2Config`, `GPT2`, `create_dataloader`, `estimate_loss`, utils functions (all already stable).
- Produces: entry scripts with stable CLI/function signatures.

- [ ] **Step 1: Remove unused imports**

- `train.py`: drop unused `import math`. Verify `os`, `time`, `torch`, `SummaryWriter` still used.
- `prepare_data.py`: verify unused imports, remove any.
- `generate.py`, `quickstart.py`: remove unused imports.

Add type hints to public helpers (`generate_text`, `interactive_mode`, `batch_generate`, `prepare_shakespeare`, `prepare_custom_data`, `prepare_openwebtext`, `minimal_training_example`, `test_model_components`) without changing behavior.

- [ ] **Step 2: Verify each script parses & CLI help works**

```bash
cd /repo && python train.py --help || true
cd /repo && python generate.py --help
cd /repo && python prepare_data.py --help
cd /repo && python quickstart.py --help
```

All should print help (or compile) without importing errors. `train.py --help` may require `argparse` — if `train()` has no argparse, just verify `python -c "import train"` succeeds.

- [ ] **Step 3: Commit**

```bash
git add train.py generate.py prepare_data.py quickstart.py
git commit -m "refactor: tidy entry scripts, remove unused imports, add type hints"
```

---

### Task 9: Component/smoke tests

**Files:**
- Create: `tests/test_components.py`
- Test: `tests/test_components.py`

**Interfaces:**
- Consumes: `GPT2Config`, `GPT2_CONFIGS`, `GPT2`, `get_tokenizer`, `CharTokenizer`, `SimpleBPETokenizer`, `get_lr`, `save_checkpoint`, `load_checkpoint`.
- Produces: pytest test functions.

- [ ] **Step 1: Write the failing tests**

```python
import torch
import pytest
from config import GPT2Config, GPT2_CONFIGS
from model import GPT2
from tokenizer import CharTokenizer, SimpleBPETokenizer, get_tokenizer
from utils import get_lr, save_checkpoint, load_checkpoint


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
    logits, loss = model(x, x)
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
```

- [ ] **Step 2: Run to see failures**

```bash
cd /repo && python -m pytest tests/test_components.py -v 2>&1 | head -40
```

Expected: import/comprehension errors surface (real API names differ from placeholders).

- [ ] **Step 3: Run the tests against the real API**

Run:

```bash
cd /repo && python -m pytest tests/test_components.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_components.py
git commit -m "test: add component/smoke tests (config, model, tokenizers, lr, checkpoint)"
```

---

### Task 10: Correctness tests

**Files:**
- Create: `tests/test_correctness.py`

**Interfaces:**
- Consumes: `GPT2`, `GPT2Config`, `DataProcessor`, `CharTokenizer`, `GPT2Tokenizer`.
- Produces: `test_causal_mask`, `test_weight_tying`, `test_topk_sampling`, `test_data_split_integrity`.

- [ ] **Step 1: Write the failing test**

```python
import os
import torch
import pytest
from config import GPT2Config
from model import GPT2
from data import DataProcessor
from tokenizer import CharTokenizer, GPT2Tokenizer


def test_causal_mask():
    cfg = GPT2Config()
    cfg.n_head = 1
    cfg.n_layer = 1
    cfg.n_embd = 32
    cfg.vocab_size = 100
    model = GPT2(cfg)
    b = model.transformer.h[0].attn.bias  # (1,1,T,T)
    T = cfg.block_size
    assert (torch.triu(b[0, 0, :T, :T], diagonal=1) == 0).all()
    assert (torch.tril(b[0, 0, :T, :T], diagonal=0) == 1).all()


def test_weight_tying():
    cfg = GPT2Config(); cfg.vocab_size=100; cfg.n_embd=32
    model = GPT2(cfg)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_topk_sampling():
    cfg = GPT2Config(); cfg.vocab_size=100; cfg.n_embd=32; cfg.n_head=2; cfg.n_layer=1
    model = GPT2(cfg).eval()
    idx = torch.randint(0, 100, (1, 8))
    gen = model.generate(idx, max_new_tokens=10, top_k=1)
    assert gen.shape[1] == 8 + 10
    # top_k=1 => greedy argmax each step; just assert same for two runs
    gen2 = model.generate(idx, max_new_tokens=10, top_k=1)
    assert torch.equal(gen, gen2)


def test_data_split_integrity(tmp_path):
    text = "hello world " * 100  # 1200 chars
    tok = CharTokenizer(); tok.train(text)
    proc = DataProcessor(tok)
    src = tmp_path / "src"; src.mkdir()
    (src / "a.txt").write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    train_n, val_n = proc.process_directory(str(src), str(out), train_ratio=0.9)
    assert train_n == int(len(text) * 0.9)
    assert val_n == len(text) - int(len(text) * 0.9)
    assert (out / "train.bin").exists() and (out / "val.bin").exists()
```

- [ ] **Step 2: Run to confirm failing**

```bash
cd /repo && python -m pytest tests/test_correctness.py -v 2>&1 | tail -30
```

Expected: failures (missing/broken pieces surfaced by the tests).

- [ ] **Step 3: Fix the code so tests pass (only if a real bug is found)**

If a bug is surfaced (e.g., mask shape, weight-tying), fix in the relevant source file. If tests already pass against current code, no source change is needed.

```bash
cd /repo && python -m pytest tests/test_correctness.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_correctness.py
git commit -m "test: add correctness tests (causal mask, weight tying, top-k, data split)"
```

---

### Task 11: End-to-end mini training test

**Files:**
- Create: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `GPT2Config`, `GPT2`, `CharTokenizer`, `DataProcessor`, `create_dataloader`.
- Produces: `test_mini_training`.

- [ ] **Step 1: Write the failing test**

```python
import torch
from config import GPT2Config
from model import GPT2
from tokenizer import CharTokenizer
from data import DataProcessor, create_dataloader


def test_mini_training(tmp_path):
    text = ("the quick brown fox jumps over the lazy dog. ") * 200
    tok = CharTokenizer(); tok.train(text)
    proc = DataProcessor(tok)
    split = int(len(text) * 0.9)
    train_f = tmp_path / "train.txt"; train_f.write_text(text[:split], encoding="utf-8")
    val_f = tmp_path / "val.txt"; val_f.write_text(text[split:], encoding="utf-8")
    data_dir = tmp_path / "data"
    proc.process_file(str(train_f), str(data_dir), "train")
    proc.process_file(str(val_f), str(data_dir), "val")

    cfg = GPT2Config()
    cfg.vocab_size = tok.vocab_size
    cfg.block_size = 64
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 32
    cfg.batch_size = 2
    cfg.max_iters = 30
    cfg.device = 'cpu'
    cfg.compile = False

    model = GPT2(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = create_dataloader(str(data_dir), "train", cfg.block_size, cfg.batch_size, "cpu")
    model.train()
    running = 0.0
    for i in range(30):
        X, Y = loader.get_batch()
        _, loss = model(X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running = float(loss.item())
    assert running < 5.0

    model.eval()
    idx = torch.tensor([tok.encode("the quick")], dtype=torch.long)
    gen = model.generate(idx, max_new_tokens=8)
    assert gen.shape[1] == idx.shape[1] + 8
    assert len(tok.decode(gen[0].tolist())) > 0
```

- [ ] **Step 2: Run to confirm failing**

```bash
cd /repo && python -m pytest tests/test_end_to_end.py -v 2>&1 | tail -30
```

Expected: failures due to placeholder API names.

- [ ] **Step 3: Fix to real API and run**

Correct the placeholder typos (e.g. `cfg.max` -> `cfg.max_iters`; confirm `loader.get_batch` and loss handling match `data.py`). Run:

```bash
cd /repo && python -m pytest tests/test_end_to_end.py -v
```

Expected: pass in ~seconds on CPU.

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: add end-to-end mini training test (train tiny GPT-2 on CPU)"
```

---

### Task 12: Run full suite + lint

**Files:**
- Modify: none (verification only)

**Interfaces:**
- Consumes: all previous tasks.

- [ ] **Step 1: Run full test suite**

```bash
cd /repo && python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff (fix trivial style issues if configured)**

```bash
cd /repo && ruff check . 2>&1 | head -40 || echo "ruff not installed"
```

If installed, fix safe findings (import order, unused imports) touched files. Do not flag-fix the pedagogical `simple_tokenizer.py` beyond the module docstring.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A && git commit -m "chore: fix lint issues" || echo "nothing to commit"
```

---

### Task 13: Rewrite `README.md`

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the corrected project structure/comands (from Tasks 1-11).

- [ ] **Step 1: Write the new README**

Rewrite in Chinese-primary with English section notes where helpful. Must match actual code: file map (add `tests/`, `requirements*.txt`, `pytest.ini`), real command flags from `prepare_data.py` / `generate.py`, checkpoint names (`checkpoint.pt`, `best_checkpoint.pt`, `checkpoint_iter_N.pt`, `final_checkpoint.pt`), config presets table, and a testing section (`pytest`). Keep the existing 参考资料 (references) section.

- [ ] **Step 2: Verify section cross-references**

```bash
cd /repo && grep -nE 'train\.py|generate\.py|prepare_data\.py|best_checkpoint|pytest' README.md
```

Confirm the commands/names actually exist in the code.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with corrected structure, setup, usage and testing"
```