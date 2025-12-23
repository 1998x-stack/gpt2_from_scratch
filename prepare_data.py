"""
Data Preparation Script
准备训练数据
"""
import os
import argparse
from tokenizer import get_tokenizer
from data import DataProcessor, prepare_shakespeare_data


def prepare_shakespeare(output_dir='./data/shakespeare'):
    """准备莎士比亚数据集（用于快速测试）"""
    print("Preparing Shakespeare dataset...")
    
    # Download and split data
    train_data, val_data = prepare_shakespeare_data('./raw_data')
    
    # Initialize tokenizer
    tokenizer = get_tokenizer('char')
    tokenizer.train(train_data + val_data)
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, 'tokenizer.pkl'))
    
    # Process and save data
    processor = DataProcessor(tokenizer)
    
    # Save train data
    train_path = os.path.join('./raw_data', 'train_temp.txt')
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_data)
    processor.process_file(train_path, output_dir, 'train')
    os.remove(train_path)
    
    # Save val data
    val_path = os.path.join('./raw_data', 'val_temp.txt')
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_data)
    processor.process_file(val_path, output_dir, 'val')
    os.remove(val_path)
    
    print(f"Shakespeare dataset prepared in {output_dir}")


def prepare_custom_data(input_path, output_dir='./data/custom', tokenizer_type='gpt2'):
    """准备自定义数据集"""
    print(f"Preparing custom dataset from {input_path}")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(tokenizer_type)
    
    # If using custom tokenizer, train it
    if tokenizer_type != 'gpt2':
        print("Training tokenizer...")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer.train(text)
        
        # Save tokenizer
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save(os.path.join(output_dir, 'tokenizer.pkl'))
    
    # Process data
    processor = DataProcessor(tokenizer)
    
    if os.path.isfile(input_path):
        # Single file: split into train/val
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]
        
        # Save splits
        os.makedirs('./temp_data', exist_ok=True)
        train_path = './temp_data/train.txt'
        val_path = './temp_data/val.txt'
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(train_text)
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write(val_text)
        
        # Process
        processor.process_file(train_path, output_dir, 'train')
        processor.process_file(val_path, output_dir, 'val')
        
        # Cleanup
        os.remove(train_path)
        os.remove(val_path)
        os.rmdir('./temp_data')
        
    elif os.path.isdir(input_path):
        # Directory: process all files
        processor.process_directory(input_path, output_dir, train_ratio=0.9)
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    print(f"Custom dataset prepared in {output_dir}")


def prepare_openwebtext(output_dir='./data/openwebtext'):
    """准备OpenWebText数据集"""
    print("Preparing OpenWebText dataset...")
    
    from data import download_openwebtext
    
    # Download dataset
    raw_dir = './raw_data/openwebtext'
    download_openwebtext(raw_dir)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer('gpt2')
    
    # Process data
    processor = DataProcessor(tokenizer)
    processor.process_file(
        os.path.join(raw_dir, 'train.txt'),
        output_dir, 'train'
    )
    processor.process_file(
        os.path.join(raw_dir, 'val.txt'),
        output_dir, 'val'
    )
    
    print(f"OpenWebText dataset prepared in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for GPT-2 training')
    parser.add_argument('--dataset', type=str, default='shakespeare',
                        choices=['shakespeare', 'openwebtext', 'custom'],
                        help='Dataset to prepare')
    parser.add_argument('--input', type=str, default=None,
                        help='Input path for custom dataset (file or directory)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        choices=['char', 'bpe', 'gpt2'],
                        help='Tokenizer type')
    
    args = parser.parse_args()
    
    if args.dataset == 'shakespeare':
        output_dir = args.output or './data/shakespeare'
        prepare_shakespeare(output_dir)
    
    elif args.dataset == 'openwebtext':
        output_dir = args.output or './data/openwebtext'
        prepare_openwebtext(output_dir)
    
    elif args.dataset == 'custom':
        if args.input is None:
            raise ValueError("Must specify --input for custom dataset")
        output_dir = args.output or './data/custom'
        prepare_custom_data(args.input, output_dir, args.tokenizer)
    
    print("\nData preparation completed!")
    print(f"You can now train the model with: python train.py")


if __name__ == '__main__':
    main()