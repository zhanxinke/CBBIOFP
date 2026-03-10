import torch
import torch.nn as nn
from torch.nn import init
from models.seqencoder import *
import torch.nn.functional as F

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.2, 1.0)
        init.constant_(m.bias.data, 0.0)
        
class CBBIOMFPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        
        self.fc1 = nn.Linear(self.args.ds.model.input_channel, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, self.args.ds.model.output_channel)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x