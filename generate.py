"""
Text Generation Script
使用训练好的模型生成文本
"""
import torch
from config import GPT2Config
from model import GPT2
from tokenizer import get_tokenizer


def generate_text(
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_k=200,
    checkpoint_path='./checkpoints/best_checkpoint.pt',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    生成文本
    
    Args:
        prompt: 输入提示文本
        max_new_tokens: 生成的最大token数
        temperature: 温度参数，越高越随机
        top_k: top-k采样参数
        checkpoint_path: 模型checkpoint路径
        device: 设备
    """
    # Load config and model
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = GPT2Config.from_dict(checkpoint['config'])
    model = GPT2(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = get_tokenizer('gpt2')
    
    # Encode prompt
    print(f"\nPrompt: {prompt}")
    print("-" * 80)
    
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            idx, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    
    print(generated_text)
    print("-" * 80)
    
    return generated_text


def interactive_mode(checkpoint_path='./checkpoints/best_checkpoint.pt'):
    """交互式生成模式"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = GPT2Config.from_dict(checkpoint['config'])
    model = GPT2(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = get_tokenizer('gpt2')
    
    print("\n" + "=" * 80)
    print("GPT-2 Interactive Generation Mode")
    print("=" * 80)
    print("Commands:")
    print("  - Enter text to generate continuation")
    print("  - Type 'quit' or 'exit' to quit")
    print("  - Type 'params' to change generation parameters")
    print("=" * 80 + "\n")
    
    # Default parameters
    temperature = 0.8
    top_k = 200
    max_new_tokens = 100
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if prompt.lower() == 'params':
                print(f"\nCurrent parameters:")
                print(f"  temperature: {temperature}")
                print(f"  top_k: {top_k}")
                print(f"  max_new_tokens: {max_new_tokens}")
                
                try:
                    temp_input = input("New temperature (press Enter to keep current): ").strip()
                    if temp_input:
                        temperature = float(temp_input)
                    
                    topk_input = input("New top_k (press Enter to keep current): ").strip()
                    if topk_input:
                        top_k = int(topk_input)
                    
                    tokens_input = input("New max_new_tokens (press Enter to keep current): ").strip()
                    if tokens_input:
                        max_new_tokens = int(tokens_input)
                    
                    print(f"\nUpdated parameters:")
                    print(f"  temperature: {temperature}")
                    print(f"  top_k: {top_k}")
                    print(f"  max_new_tokens: {max_new_tokens}")
                except ValueError:
                    print("Invalid input, parameters unchanged.")
                
                continue
            
            if not prompt:
                continue
            
            # Encode and generate
            encoded = tokenizer.encode(prompt)
            idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
            
            print("\nGenerating...")
            with torch.no_grad():
                generated = model.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            
            generated_text = tokenizer.decode(generated[0].tolist())
            
            print("\n" + "-" * 80)
            print(generated_text)
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_generate(
    prompts,
    checkpoint_path='./checkpoints/best_checkpoint.pt',
    output_file='generated_samples.txt',
    max_new_tokens=100,
    temperature=0.8,
    top_k=200
):
    """批量生成文本"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPT2Config.from_dict(checkpoint['config'])
    model = GPT2(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    tokenizer = get_tokenizer('gpt2')
    
    results = []
    
    print(f"Generating {len(prompts)} samples...")
    for i, prompt in enumerate(prompts, 1):
        print(f"Sample {i}/{len(prompts)}: {prompt[:50]}...")
        
        encoded = tokenizer.encode(prompt)
        idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        results.append(generated_text)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(results, 1):
            f.write(f"=== Sample {i} ===\n")
            f.write(text)
            f.write("\n\n")
    
    print(f"Saved {len(results)} samples to {output_file}")
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate text with GPT-2')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Prompt text')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top-k for sampling')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_checkpoint.pt',
                        help='Path to checkpoint')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.checkpoint)
    else:
        generate_text(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            checkpoint_path=args.checkpoint
        )