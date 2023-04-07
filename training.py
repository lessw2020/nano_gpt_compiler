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

# --- globals
_valid_models = ["dev", "mini"]


def training_main():
    print(f"Welcome! ")


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch experiments with NanoGPT and DistCompiler"
    )
    parser.add_argument(
        "--model",
        default="mini",
        metavar="string",
        choices=_valid_models,
        help="choose NanoGPT model to run, available: `mini`, `dev`, (default: mini)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    curr_model = args.model
    print(f"===> Loading Model: {args.model=}")
    assert (
        curr_model in _valid_models
    ), f"Model {curr_model} is not supported. Check valid list in {__name__}. Aborting..."

    if curr_model in ["dev", "mini"]:
        import config.shakespeare_config as config

    training_main()
