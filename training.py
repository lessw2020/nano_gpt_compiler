import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import argparse
import os
import time

import colorama
import torch

from colorama import Fore

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)

# import model_checkpointing

import torch.distributed as dist

# import environment

# bf16_ready = environment.verify_bfloat_support

from torch.utils.data import DistributedSampler

# prep -------
colorama.init(autoreset=True)  # reset after every line
