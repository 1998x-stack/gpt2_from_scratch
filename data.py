"""
Data Loading and Processing Pipeline
支持多种数据集和高效的数据加载
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
        return x, y


class DataLoaderWrapper:
    """
    数据加载器包装类
    参考nanoGPT的设计，使用memory-mapped arrays提高效率
    """
    def __init__(self, data_dir, split, block_size, batch_size, device='cpu'):
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # 加载数据
        data_path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
    def get_batch(self):
        """获取一个batch的数据"""
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if self.device == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        
        return x, y


class DataProcessor:
    """数据预处理和保存"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def process_file(self, input_path, output_dir, split='train'):
        """
        处理单个文本文件
        将文本tokenize并保存为二进制文件
        """
        print(f"Processing {input_path}...")
        
        # 读取文本
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        tokens_np = np.array(tokens, dtype=np.uint16)
        
        # 保存
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{split}.bin')
        tokens_np.tofile(output_path)
        
        print(f"Saved {len(tokens)} tokens to {output_path}")
        return len(tokens)
    
    def process_directory(self, input_dir, output_dir, train_ratio=0.9):
        """
        处理目录中的所有文本文件
        自动分割训练集和验证集
        """
        all_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        all_text = []
        
        for fname in all_files:
            fpath = os.path.join(input_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                all_text.append(f.read())
        
        # 合并所有文本
        full_text = '\n'.join(all_text)
        
        # 分割训练集和验证集
        split_idx = int(len(full_text) * train_ratio)
        train_text = full_text[:split_idx]
        val_text = full_text[split_idx:]
        
        # Tokenize and save
        print("Tokenizing training data...")
        train_tokens = self.tokenizer.encode(train_text)
        train_tokens_np = np.array(train_tokens, dtype=np.uint16)
        
        print("Tokenizing validation data...")
        val_tokens = self.tokenizer.encode(val_text)
        val_tokens_np = np.array(val_tokens, dtype=np.uint16)
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        train_tokens_np.tofile(os.path.join(output_dir, 'train.bin'))
        val_tokens_np.tofile(os.path.join(output_dir, 'val.bin'))
        
        print(f"Train tokens: {len(train_tokens)}")
        print(f"Val tokens: {len(val_tokens)}")
        
        return len(train_tokens), len(val_tokens)


def download_openwebtext(data_dir):
    """
    下载OpenWebText数据集
    这是一个示例函数，实际使用时需要安装datasets库
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要安装 datasets 库: pip install datasets")
        return
    
    print("Downloading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", num_proc=8)
    
    # 分割数据集
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    
    # 保存为文本文件
    def save_split(split, fname):
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            for example in split:
                f.write(example['text'] + '\n')
        print(f"Saved {len(split)} examples to {fpath}")
    
    os.makedirs(data_dir, exist_ok=True)
    save_split(split_dataset['train'], 'train.txt')
    save_split(split_dataset['val'], 'val.txt')


def prepare_shakespeare_data(data_dir):
    """
    准备莎士比亚数据集（小型测试数据）
    """
    import urllib.request
    
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    fpath = os.path.join(data_dir, 'shakespeare.txt')
    
    if not os.path.exists(fpath):
        print(f"Downloading Shakespeare dataset from {url}")
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, fpath)
    
    # 读取数据
    with open(fpath, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # 分割训练集和验证集
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    return train_data, val_data


def create_dataloader(data_dir, split, block_size, batch_size, device='cpu'):
    """创建数据加载器"""
    return DataLoaderWrapper(data_dir, split, block_size, batch_size, device)


def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    """评估模型在训练集和验证集上的loss"""
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