"""Drive train.py on the Grimms sample corpus with a small, CPU-friendly config."""
from config import GPT2Config

# Override class attributes so train.py's GPT2Config() picks them up.
GPT2Config.vocab_size = 89      # char-level vocab from tokenizer.pkl
GPT2Config.block_size = 256
GPT2Config.n_layer = 4
GPT2Config.n_head = 4
GPT2Config.n_embd = 128
GPT2Config.batch_size = 16
GPT2Config.dropout = 0.1
GPT2Config.max_iters = 3000
GPT2Config.learning_rate = 6e-4
GPT2Config.warmup_iters = 200
GPT2Config.lr_decay_iters = 3000
GPT2Config.min_lr = 6e-5
GPT2Config.eval_interval = 300
GPT2Config.eval_iters = 50
GPT2Config.log_interval = 50
GPT2Config.save_interval = 1000
GPT2Config.checkpoint_dir = './checkpoints_grimms'
GPT2Config.log_dir = './runs_grimms'
GPT2Config.data_dir = './data/grimms'
GPT2Config.device = 'cpu'
GPT2Config.compile = False

import train

train.train()