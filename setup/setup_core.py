import torch

def seed_init(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    