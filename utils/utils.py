import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self): 
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=None)
        
    def forward(self, pred, label):
        return self.loss_fn(pred, label)
    