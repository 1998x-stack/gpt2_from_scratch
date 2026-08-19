"""Data loading and light preprocessing.

Encodes text (via an injected tokenizer) into ``uint16`` ``.bin`` files and
provides a nanoGPT-style memory-mapped :class:`DataLoaderWrapper`.
加载数据：二进制 memmap 读取，训练时每次随机采样一个 batch。
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch


class DataLoaderWrapper:
    """
    A nanoGPT-style batch sampler over a memory-mapped ``.bin`` file.

    Uses ``np.memmap`` so arbitrarily large datasets are handled without
    loading the full array into RAM.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        block_size: int,
        batch_size: int,
        device: str = 'cpu',
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        data_path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one batch of (inputs, targets) of ``batch_size x block_size``."""
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack(
            [torch.from_numpy((self.data[i:i + self.block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (self.data[i + 1:i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )

        if self.device == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


class DataProcessor:
    """Tokenize text files and write them as ``uint16`` ``.bin`` files."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def process_file(self, input_path: str, output_dir: str, split: str = 'train') -> int:
        """Tokenize a single text file and save it as ``{split}.bin``.

        Returns:
            The number of tokens written.
        """
        print(f"Processing {input_path}...")

        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = self.tokenizer.encode(text)
        tokens_np = np.array(tokens, dtype=np.uint16)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{split}.bin')
        tokens_np.tofile(output_path)

        print(f"Saved {len(tokens)} tokens to {output_path}")
        return len(tokens)

    def process_directory(self, input_dir: str, output_dir: str, train_ratio: float = 0.9) -> Tuple[int, int]:
        """Tokenize all ``.txt`` files in ``input_dir`` into train/val ``.bin`` files.

        Returns:
            (num_train_tokens, num_val_tokens).
        """
        all_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        all_text = []

        for fname in all_files:
            fpath = os.path.join(input_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                all_text.append(f.read())

        full_text = '\n'.join(all_text)

        split_idx = int(len(full_text) * train_ratio)
        train_text = full_text[:split_idx]
        val_text = full_text[split_idx:]

        print("Tokenizing training data...")
        train_tokens = self.tokenizer.encode(train_text)
        train_tokens_np = np.array(train_tokens, dtype=np.uint16)

        print("Tokenizing validation data...")
        val_tokens = self.tokenizer.encode(val_text)
        val_tokens_np = np.array(val_tokens, dtype=np.uint16)

        os.makedirs(output_dir, exist_ok=True)
        train_tokens_np.tofile(os.path.join(output_dir, 'train.bin'))
        val_tokens_np.tofile(os.path.join(output_dir, 'val.bin'))

        print(f"Train tokens: {len(train_tokens)}")
        print(f"Val tokens: {len(val_tokens)}")

        return len(train_tokens), len(val_tokens)


def download_openwebtext(data_dir: str) -> None:
    """
    Download the OpenWebText dataset (requires the ``datasets`` library).

    Example helper. Callers should ``pip install datasets`` first.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要安装 datasets 库: pip install datasets")
        return

    print("Downloading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", num_proc=8)

    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def save_split(split, fname):
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            for example in split:
                f.write(example['text'] + '\n')
        print(f"Saved {len(split)} examples to {fpath}")

    os.makedirs(data_dir, exist_ok=True)
    save_split(split_dataset['train'], 'train.txt')
    save_split(split_dataset['val'], 'val.txt')


def prepare_shakespeare_data(data_dir: str) -> Tuple[str, str]:
    """Download the tiny Shakespeare dataset and split it into train/val texts."""
    import urllib.request

    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    fpath = os.path.join(data_dir, 'shakespeare.txt')

    if not os.path.exists(fpath):
        print(f"Downloading Shakespeare dataset from {url}")
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, fpath)

    with open(fpath, 'r', encoding='utf-8') as f:
        data = f.read()

    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    return train_data, val_data


def create_dataloader(data_dir: str, split: str, block_size: int, batch_size: int, device: str = 'cpu') -> DataLoaderWrapper:
    """Create a :class:`DataLoaderWrapper` for the given split."""
    return DataLoaderWrapper(data_dir, split, block_size, batch_size, device)


def estimate_loss(model, train_loader, val_loader, eval_iters: int, device) -> dict:
    """Estimate mean loss over ``eval_iters`` batches on both splits."""
    out = {}
    model.eval()

    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            with torch.no_grad():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out