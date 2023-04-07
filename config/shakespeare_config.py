import time
from dataclasses import dataclass
from typing import Tuple
import os

import torch
import torchvision.transforms as transforms
import tqdm
from torch import distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class train_config:
    out_dir = "out-shakespeare-char"
    eval_interval = 250  # keep frequent because we'll overfit
    eval_iters = 200
    log_interval = 10  # don't print too too often

    # we expect to overfit on this small dataset, so only save when val improves
    always_save_checkpoint = False

    dataset = "shakespeare_char"
    training_batch_size = 9
    block_size = 256  # context of up to 256 previous characters

    # baby GPT model :)

    n_layer = 2
    n_head = 2
    n_embd = 192  # 768 // 2
    dropout = 0.0

    learning_rate = 1e-3  # with baby networks can afford to go a bit higher
    max_iters = 20
    lr_decay_iters = 5000  # make equal to max_iters usually
    min_lr = 1e-4  # learning_rate / 10 usually
    beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

    warmup_iters = 2  # not super necessary potentially
    save_optimizer: bool = False
    load_optimizer: bool = False
