import torch
import numpy as np
from ops.load_data import load_data



class UEDNetDataset():
    def __init__(self, data, batch_size,Shuffle=True):
        self.data = data
        X = torch.tensor(self.data, dtype=torch.float32)
        self.dataset = torch.utils.data.TensorDataset(X,X)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size = batch_size, shuffle=Shuffle)


