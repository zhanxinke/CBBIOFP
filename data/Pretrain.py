import os
from pandas.core.config_init import data_manager_doc
import torch
import pandas as pd
from loguru import logger
from utils.utils import *
from hydra.utils import get_original_cwd

# 20 种标准氨基酸（顺序很重要，用 string）
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA2ID = {aa: i + 1 for i, aa in enumerate(STANDARD_AA)}
    
def normalize_aa(seq):
    return "".join([aa for aa in seq if aa in AA2ID])

def encode_fixed_length(seq, MAX_LEN=50, PAD_ID=0):
    seq = normalize_aa(seq)
    ids = [AA2ID[aa] for aa in seq]

    if len(ids) >= MAX_LEN:
        ids = ids[:MAX_LEN]
    else:
        ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return ids


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, name, split="pretrain"):
        super().__init__()
        self.name = name
        self.split = split
        
        self.data_path = get_original_cwd() + '/data/' + self.name 

        self.processed_path = os.path.join(self.data_path, self.split + ".pt")

        if not os.path.isfile(self.processed_path):
            logger.info("Pre-processed dataset not found, Start pre-processing...")
            ori_data = os.path.join(self.data_path, self.split + ".txt")
            self.process(ori_data, self.split)
        else:
            logger.info(f"loading pre-processed dataset from {self.split}...")

        self.data = torch.load(self.processed_path)

    def process(self, data_file, split_name):
        
        data_list = []
        label_list = []

        with open(data_file, "r") as f:
            cur_label = None
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    cur_label = line[1:].split()[0]  # 例如 '0000100...'
                    continue

                # 序列行
                ids = encode_fixed_length(line)  # list[int], 长度 L（比如 50）
                x = torch.tensor(ids, dtype=torch.long)

                # 标签：多标签 0/1 串 -> FloatTensor
                assert cur_label is not None, "序列行前必须先有 >label 行"
                y = torch.tensor([int(c) for c in cur_label], dtype=torch.float)

                data_list.append(x)
                label_list.append(1)

        assert len(data_list) == len(label_list), (len(data_list), len(label_list))
        dataset = list(zip(data_list, label_list))

        torch.save(dataset, os.path.join(self.data_path, f"{split_name}.pt"))

    def __len__(self):
        return len(self.data)   # ✅ 不变

    def __getitem__(self, index):
        return self.data[index] # ✅ 不变
    
    