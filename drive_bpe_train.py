"""Drive train.py on the Grimms BPE dataset (vocab=1024) with a small CPU model."""
from config import GPT2Config

GPT2Config.vocab_size = 1024    # byte-level BPE vocab
GPT2Config.block_size = 256
GPT2Config.n_layer = 4
GPT2Config.n_head = 8
GPT2Config.n_embd = 256
GPT2Config.batch_size = 16
GPT2Config.dropout = 0.1
GPT2Config.max_iters = 1500
GPT2Config.learning_rate = 6e-4
GPT2Config.warmup_iters = 100
GPT2Config.lr_decay_iters = 1500
GPT2Config.min_lr = 6e-5
GPT2Config.eval_interval = 150
GPT2Config.eval_iters = 50
GPT2Config.log_interval = 50
GPT2Config.save_interval = 750
GPT2Config.checkpoint_dir = './checkpoints_grimms_bpe'
GPT2Config.log_dir = './runs_grimms_bpe'
GPT2Config.data_dir = './data/grimms_bpe'
GPT2Config.device = 'cpu'
GPT2Config.compile = False

import train

train.train()