import torch.utils.data as data
import random
import numpy as np
import torch

class Folder(data.Dataset):

    def __init__(self):
        train_feature = torch.load("/home/xinke/Projects/MultiFunctional_Peptides/MFTPCBB/results/MFTP/fold1_batch64_epoch200/train_feature_graph.pt")
        test_feature = torch.load("/home/xinke/Projects/MultiFunctional_Peptides/MFTPCBB/results/MFTP/fold1_batch64_epoch200/test_feature_graph.pt")

        self.train_seq = train_feature['peptide']
        self.train_data = train_feature['feature']
        self.train_label = train_feature['label']

        self.test_seq = test_feature['peptide']
        self.test_data = test_feature['feature']
        self.test_label = test_feature['label']
        

class Dataset(data.Dataset): 
    def __init__(self, seq, data, label):
        self.seq = seq
        self.data = data
        self.label = label

    def __len__(self):
        length = len(self.label)
        return length

    def __getitem__(self, index):
        return self.seq[index], self.data[index], self.label[index]



if __name__ == '__main__':
     print("1")

