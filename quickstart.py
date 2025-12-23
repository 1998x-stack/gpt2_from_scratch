"""
Quickstart Example
快速开始使用 GPT-2 训练和生成
"""
import os
import torch
from config import GPT2Config
from model import GPT2
from tokenizer import get_tokenizer
from data import prepare_shakespeare_data, DataProcessor, create_dataloader
from utils import save_checkpoint, print_training_info


def minimal_training_example():
    """
    最小化训练示例
    使用莎士比亚数据集进行快速演示
    """
    print("=" * 80)
    print("Minimal GPT-2 Training Example")
    print("=" * 80)
    
    # 1. 准备数据
    print("\n1. Preparing data...")
    data_dir = './data/shakespeare_mini'
    os.makedirs(data_dir, exist_ok=True)
    
    train_text, val_text = prepare_shakespeare_data('./raw_data')
    
    # 使用字符级 tokenizer（更简单）
    tokenizer = get_tokenizer('char')
    tokenizer.train(train_text + val_text)
    
    processor = DataProcessor(tokenizer)
    
    # 保存临时文件
    with open('/tmp/train.txt', 'w') as f:
        f.write(train_text)
    with open('/tmp/val.txt', 'w') as f:
        f.write(val_text)
    
    processor.process_file('/tmp/train.txt', data_dir, 'train')
    processor.process_file('/tmp/val.txt', data_dir, 'val')
    
    # 2. 配置模型（小模型，快速训练）
    print("\n2. Configuring model...")
    config = GPT2Config()
    config.vocab_size = tokenizer.vocab_size
    config.block_size = 256      # 较小的上下文
    config.n_layer = 6           # 较少的层数
    config.n_head = 6            # 较少的注意力头
    config.n_embd = 384          # 较小的嵌入维度
    config.batch_size = 64       # 较大的 batch
    config.max_iters = 5000      # 较少的迭代次数
    config.eval_interval = 500
    config.log_interval = 100
    config.data_dir = data_dir
    
    # 3. 初始化模型
    print("\n3. Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = device
    
    model = GPT2(config)
    model.to(device)
    
    print_training_info(config, model, len(train_text), len(val_text))
    
    # 4. 准备数据加载器
    print("\n4. Preparing data loaders...")
    train_loader = create_dataloader(
        data_dir, 'train', config.block_size, config.batch_size, device
    )
    
    # 5. 简单训练循环
    print("\n5. Starting training...")
    print("-" * 80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    
    for iter_num in range(config.max_iters):
        # 获取数据
        X, Y = train_loader.get_batch()
        
        # 前向传播
        logits, loss = model(X, Y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 日志
        if iter_num % config.log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}")
    
    # 6. 保存模型
    print("\n6. Saving model...")
    config.checkpoint_dir = './checkpoints_mini'
    save_checkpoint(model, optimizer, config.max_iters, loss.item(), config)
    
    # 7. 生成示例
    print("\n7. Generating text...")
    print("-" * 80)
    
    model.eval()
    prompt = "ROMEO:"
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=200, temperature=0.8)
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(generated_text)
    print("-" * 80)
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {config.checkpoint_dir}/checkpoint.pt")


def test_model_components():
    """测试模型各个组件"""
    print("\n" + "=" * 80)
    print("Testing Model Components")
    print("=" * 80)
    
    # 测试配置
    print("\n1. Testing Configuration...")
    config = GPT2Config()
    print(f"   ✓ Config created: {config.n_layer} layers, {config.n_embd} dims")
    
    # 测试模型
    print("\n2. Testing Model...")
    model = GPT2(config)
    n_params = model.get_num_params()
    print(f"   ✓ Model created: {n_params:,} parameters")
    
    # 测试前向传播
    print("\n3. Testing Forward Pass...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    batch_size = 4
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    logits, loss = model(x, y)
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Logits shape: {logits.shape}")
    print(f"   ✓ Loss: {loss.item():.4f}")
    
    # 测试生成
    print("\n4. Testing Generation...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(x[:1, :10], max_new_tokens=20)
    print(f"   ✓ Generation successful")
    print(f"   ✓ Generated shape: {generated.shape}")
    
    # 测试 tokenizer
    print("\n5. Testing Tokenizer...")
    tokenizer = get_tokenizer('char')
    test_text = "Hello, World!"
    tokenizer.train(test_text)
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"   ✓ Original: {test_text}")
    print(f"   ✓ Encoded: {encoded}")
    print(f"   ✓ Decoded: {decoded}")
    assert test_text == decoded, "Tokenizer encode/decode mismatch!"
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Run training example or component tests')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        minimal_training_example()
    else:
        test_model_components()