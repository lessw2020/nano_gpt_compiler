import torch
import torch.distributed as dist


def seed_init(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def start_backend():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group("nccl")


def set_device(local_rank: int):
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)


def cleanup(logger, rank):
    dist.barrier()
    if rank == 0:
        logger.warning(f"Logging off...")
    dist.destroy_process_group()
