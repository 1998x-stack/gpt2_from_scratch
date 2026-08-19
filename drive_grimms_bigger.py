"""Drive train.py on the Grimms sample corpus with a larger model for better output."""
from config import GPT2Config

# Override class attributes so train.py's GPT2Config() picks them up.
GPT2Config.vocab_size = 89      # char-level vocab from tokenizer.pkl
GPT2Config.block_size = 256
GPT2Config.n_layer = 6
GPT2Config.n_head = 8
GPT2Config.n_embd = 256
GPT2Config.batch_size = 16
GPT2Config.dropout = 0.1
GPT2Config.max_iters = 2000
GPT2Config.learning_rate = 6e-4
GPT2Config.warmup_iters = 150
GPT2Config.lr_decay_iters = 2000
GPT2Config.min_lr = 6e-5
GPT2Config.eval_interval = 200
GPT2Config.eval_iters = 50
GPT2Config.log_interval = 50
GPT2Config.save_interval = 1000
GPT2Config.checkpoint_dir = './checkpoints_grimms2'
GPT2Config.log_dir = './runs_grimms2'
GPT2Config.data_dir = './data/grimms'
GPT2Config.device = 'cpu'
GPT2Config.compile = False

import train

train.train()