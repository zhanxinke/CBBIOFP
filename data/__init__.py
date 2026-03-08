import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils.utils import *
from data.MFTP import MFTPDataset
from data.Pretrain import PretrainDataset

def load_metric(args):
    if args.ds.basic.dataset in ['MFTP', 'pretrain']:
        return MAE
    else:
        raise Exception('Unknown dataset!')


def get_loss_fn(args):
    if args.ds.basic.dataset in ['MFTP', 'pretrain']:
        loss_fn = CELoss()
        
    else:
        raise Exception('Unknown dataset!')
    return loss_fn


def load_data(args):
    if args.ds.basic.dataset in ['MFTP']:
        train_data = MFTPDataset(args.ds.basic.dataset, 'train')
        test_data = MFTPDataset(args.ds.basic.dataset, 'test')
        return train_data, test_data
    elif args.ds.basic.dataset in ['pretrain']:
        train_data = PretrainDataset(args.ds.basic.dataset, 'pretrain_train')
        test_data = PretrainDataset(args.ds.basic.dataset, 'pretrain_val')
        return train_data, test_data
    else:
        raise Exception('Unknown dataset!')


def set_seed(seed: int, deterministic: bool = True):
    """
    Args:
        seed (int): Random seed
        deterministic (bool): Whether to enable deterministic mode (slightly affects performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.enabled = True

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
        
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path