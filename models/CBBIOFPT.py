import torch
import torch.nn as nn
from models.seqencoder import *

class CBBIOMFPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
        self.fc1 = nn.Linear(self.args.ds.model.input_channel, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, self.args.ds.model.output_channel)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x