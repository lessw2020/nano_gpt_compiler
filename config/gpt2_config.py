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
    seed: int = 2023

    dtype = torch.bfloat16

    out_dir = "out"
    eval_interval = 2000
    log_interval = 1
    eval_iters = 200
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval
    init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

    # data
    dataset = "openwebtext"
    gradient_accumulation_steps = 5  # used to simulate larger batch sizes
    batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 1024
    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias = True  # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 6e-4  # max learning rate
    max_iters = 600000  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 2000  # how many steps to warm up for
    lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    dynamo_compile = True  # use PyTorch 2.0 to compile the model to be faster
