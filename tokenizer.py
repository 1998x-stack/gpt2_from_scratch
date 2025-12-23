"""
Simple Tokenizer for GPT-2
使用字符级或简单的BPE tokenization
"""
import os
import pickle
import regex as re
from collections import Counter


class CharTokenizer:
    """简单的字符级tokenizer"""
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
    
    def train(self, text):
        """从文本训练tokenizer"""
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inverse_vocab = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text):
        """编码文本为token IDs"""
        return [self.vocab[ch] for ch in text if ch in self.vocab]
    
    def decode(self, tokens):
        """解码token IDs为文本"""
        return ''.join([self.inverse_vocab[t] for t in tokens])
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def save(self, path):
        """保存tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({'vocab': self.vocab, 'inverse_vocab': self.inverse_vocab}, f)
    
    def load(self, path):
        """加载tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.inverse_vocab = data['inverse_vocab']


class SimpleBPETokenizer:
    """简化的BPE tokenizer (类似GPT-2)"""
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        
    def get_stats(self, ids):
        """统计相邻token对的频率"""
        counts = Counter()
        for pair in zip(ids[:-1], ids[1:]):
            counts[pair] += 1
        return counts
    
    def merge(self, ids, pair, idx):
        """合并token对"""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, text, num_merges=10000):
        """训练BPE tokenizer"""
        # 初始化：每个字节作为一个token
        tokens = list(text.encode('utf-8'))
        ids = list(tokens)
        
        # 基础vocab: 0-255 字节
        vocab = {i: bytes([i]) for i in range(256)}
        
        # 迭代合并最频繁的pair
        merges = {}
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            if (i + 1) % 1000 == 0:
                print(f"Merge {i+1}/{num_merges}")
        
        # 构建编码器和解码器
        self.encoder = {v: k for k, v in vocab.items()}
        self.decoder = vocab
        self.bpe_ranks = merges
    
    def encode(self, text):
        """编码文本"""
        tokens = list(text.encode('utf-8'))
        
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            if pair not in self.bpe_ranks:
                break
            idx = self.bpe_ranks[pair]
            tokens = self.merge(tokens, pair, idx)
        
        return tokens
    
    def decode(self, tokens):
        """解码tokens"""
        bytes_list = [self.decoder[t] for t in tokens]
        text_bytes = b''.join(bytes_list)
        return text_bytes.decode('utf-8', errors='replace')
    
    def save(self, path):
        """保存tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'encoder': self.encoder,
                'decoder': self.decoder,
                'bpe_ranks': self.bpe_ranks
            }, f)
    
    def load(self, path):
        """加载tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.encoder = data['encoder']
            self.decoder = data['decoder']
            self.bpe_ranks = data['bpe_ranks']


class GPT2Tokenizer:
    """
    使用预训练的GPT-2 tokenizer (需要tiktoken或transformers)
    这是推荐的方式，因为GPT-2使用特定的BPE vocab
    """
    def __init__(self):
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.backend = 'tiktoken'
        except ImportError:
            try:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                self.backend = 'transformers'
            except ImportError:
                raise ImportError("需要安装 tiktoken 或 transformers: pip install tiktoken")
    
    def encode(self, text):
        """编码文本"""
        if self.backend == 'tiktoken':
            return self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        else:
            return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        """解码tokens"""
        return self.tokenizer.decode(tokens)
    
    @property
    def vocab_size(self):
        if self.backend == 'tiktoken':
            return self.tokenizer.n_vocab
        else:
            return len(self.tokenizer)
    
    @property
    def eot_token(self):
        """End of text token"""
        if self.backend == 'tiktoken':
            return self.tokenizer.eot_token
        else:
            return self.tokenizer.eos_token_id


def get_tokenizer(tokenizer_type='gpt2'):
    """获取tokenizer实例"""
    if tokenizer_type == 'char':
        return CharTokenizer()
    elif tokenizer_type == 'bpe':
        return SimpleBPETokenizer()
    elif tokenizer_type == 'gpt2':
        return GPT2Tokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")