import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTM_AutoEncoder(nn):
    def __init__(self, in_dim, hidden_size):
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.encoder_LSTM = nn.
