import random
import numpy as np
import torch
import transformers

def seed_everything(seed):
    """全ての乱数シードを固定する関数"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
    print(f"乱数シードを {seed} に固定しました。")

def seed_worker(worker_id):
    """DataLoaderの各ワーカーのシードを設定する関数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)