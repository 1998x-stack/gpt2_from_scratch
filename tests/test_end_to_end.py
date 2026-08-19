"""End-to-end test: train a tiny GPT-2 on CPU and generate text."""
import torch

from config import GPT2Config
from model import GPT2
from tokenizer import CharTokenizer
from data import DataProcessor, create_dataloader


def test_mini_training(tmp_path):
    text = ("the quick brown fox jumps over the lazy dog. ") * 200
    tok = CharTokenizer()
    tok.train(text)
    proc = DataProcessor(tok)

    split = int(len(text) * 0.9)
    train_f = tmp_path / "train.txt"
    train_f.write_text(text[:split], encoding="utf-8")
    val_f = tmp_path / "val.txt"
    val_f.write_text(text[split:], encoding="utf-8")
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
    last_loss = None
    for _ in range(30):
        X, Y = loader.get_batch()
        _, loss = model(X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    assert last_loss is not None and last_loss < 5.0

    model.eval()
    idx = torch.tensor([tok.encode("the quick")], dtype=torch.long)
    gen = model.generate(idx, max_new_tokens=8)
    assert gen.shape[1] == idx.shape[1] + 8
    assert len(tok.decode(gen[0].tolist())) > 0