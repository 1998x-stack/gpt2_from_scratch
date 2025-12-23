"""
Training Script for GPT-2
参考nanoGPT的训练流程
"""
import os
import time
import math
import torch
from torch.utils.tensorboard import SummaryWriter

from config import GPT2Config
from model import GPT2
from data import create_dataloader, estimate_loss
from utils import (
    get_lr, save_checkpoint, load_checkpoint, 
    print_training_info, setup_logging, 
    get_device_context, clip_gradients, set_seed,
    save_training_config
)


def train():
    # ==================== Configuration ====================
    config = GPT2Config()
    
    # Setup
    setup_logging(config)
    set_seed(42)
    save_training_config(config)
    
    # Device
    device = config.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    print(f"Using device: {device}")
    
    # ==================== Data ====================
    print("Loading data...")
    train_loader = create_dataloader(
        config.data_dir, 'train', 
        config.block_size, config.batch_size, device
    )
    val_loader = create_dataloader(
        config.data_dir, 'val', 
        config.block_size, config.batch_size, device
    )
    
    # ==================== Model ====================
    print("Initializing model...")
    model = GPT2(config)
    model.to(device)
    
    # Print model info
    n_params = model.get_num_params()
    print_training_info(
        config, model, 
        len(train_loader.data), 
        len(val_loader.data)
    )
    
    # Compile model (PyTorch 2.0+)
    if config.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # ==================== Optimizer ====================
    optimizer = model.configure_optimizers(
        config.weight_decay, 
        config.learning_rate,
        (config.beta1, config.beta2),
        device_type
    )
    
    # ==================== Resume from checkpoint ====================
    iter_num = 0
    best_val_loss = float('inf')
    
    checkpoint_path = os.path.join(config.checkpoint_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        iter_num, best_val_loss = load_checkpoint(model, optimizer, config)
    
    # ==================== Tensorboard ====================
    writer = SummaryWriter(config.log_dir)
    
    # ==================== Training Loop ====================
    print("\nStarting training...")
    print("=" * 80)
    
    X, Y = train_loader.get_batch()  # Warm up
    t0 = time.time()
    running_loss = 0.0
    
    while iter_num < config.max_iters:
        # ==================== Learning Rate Schedule ====================
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # ==================== Evaluation ====================
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, config.eval_iters, device)
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Tensorboard logging
            writer.add_scalar('Loss/train', losses['train'], iter_num)
            writer.add_scalar('Loss/val', losses['val'], iter_num)
            writer.add_scalar('Learning_rate', lr, iter_num)
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpoint(
                        model, optimizer, iter_num, best_val_loss, 
                        config, 'best_checkpoint.pt'
                    )
        
        # ==================== Training Step ====================
        model.train()
        
        # Get batch
        X, Y = train_loader.get_batch()
        
        # Forward pass with mixed precision
        ctx = get_device_context(device_type, config.dtype)
        with ctx:
            logits, loss = model(X, Y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip != 0.0:
            clip_gradients(model, config.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # ==================== Logging ====================
        running_loss += loss.item()
        
        if iter_num % config.log_interval == 0 and iter_num > 0:
            t1 = time.time()
            dt = t1 - t0
            lossf = running_loss / config.log_interval
            
            # Calculate tokens per second
            tokens_per_iter = config.batch_size * config.block_size
            tokens_per_sec = tokens_per_iter * config.log_interval / dt
            
            print(f"iter {iter_num}: loss {lossf:.4f}, "
                  f"time {dt*1000:.2f}ms, "
                  f"tok/sec {tokens_per_sec:.2f}, "
                  f"lr {lr:.2e}")
            
            # Tensorboard
            writer.add_scalar('Loss/train_running', lossf, iter_num)
            writer.add_scalar('Performance/tokens_per_sec', tokens_per_sec, iter_num)
            
            running_loss = 0.0
            t0 = time.time()
        
        # ==================== Checkpointing ====================
        if iter_num % config.save_interval == 0 and iter_num > 0:
            save_checkpoint(
                model, optimizer, iter_num, best_val_loss, 
                config, f'checkpoint_iter_{iter_num}.pt'
            )
        
        iter_num += 1
    
    # ==================== Final Checkpoint ====================
    print("\nTraining completed!")
    save_checkpoint(
        model, optimizer, iter_num, best_val_loss, 
        config, 'final_checkpoint.pt'
    )
    
    writer.close()
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    train()