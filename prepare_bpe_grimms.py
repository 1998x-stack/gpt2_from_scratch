"""Prepare a compact byte-level BPE dataset from sample_corpus.txt.

Trains a 1024-token BPE (tokenizer.py) on the corpus and writes
train.bin / val.bin / tokenizer.pkl under ./data/grimms_bpe.
"""
import os

import numpy as np
from tokenizer import SimpleBPETokenizer

VOCAB_SIZE = 1024
OUT = './data/grimms_bpe'
SRC = 'sample_corpus.txt'


def main() -> None:
    text = open(SRC, encoding='utf-8').read()
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]

    tok = SimpleBPETokenizer(vocab_size=VOCAB_SIZE)
    print(f"Training byte-level BPE (vocab={VOCAB_SIZE}) on {len(train_text)} chars...")
    tok.train(train_text, num_merges=VOCAB_SIZE - 256)

    os.makedirs(OUT, exist_ok=True)
    tok.save(os.path.join(OUT, 'tokenizer.pkl'))

    for name, chunk in [('train', train_text), ('val', val_text)]:
        ids = np.array(tok.encode(chunk), dtype=np.uint16)
        ids.tofile(os.path.join(OUT, f'{name}.bin'))
        print(f"{name}: {len(ids)} tokens -> {OUT}/{name}.bin")

    print(f"Vocab size: {len(tok.decoder)}")


if __name__ == '__main__':
    main()