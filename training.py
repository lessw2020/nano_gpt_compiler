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

from setup import setup_core

# prep -------
colorama.init(autoreset=True)  # reset after every line

# --- globals
_valid_models = ["dev", "mini"]

import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.setLevel = logging.INFO


def _log(msg):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logger.warning(f"{msg}")


def get_environment():
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    return (rank, global_rank, world_size)


def training_main(model_name):
    train_cfg = config.train_config()  # loads from defaults

    # seed
    setup_core.seed_init(train_cfg.seed)

    local_rank, global_rank, world_size = get_environment()

    setup_core.start_backend()
    setup_core.set_device(local_rank)
    _log(f"Running model: {model_name}")
    _log(f"World Environment:\n{world_size=}\n")

    setup_core.cleanup(logger, local_rank)


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
    assert (
        curr_model in _valid_models
    ), f"Model {curr_model} is not supported. Check valid list in {__name__}. Aborting..."

    if curr_model in ["dev", "mini"]:
        import config.shakespeare_config as config

    training_main(curr_model)
