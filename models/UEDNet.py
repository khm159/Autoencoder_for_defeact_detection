import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

class UEDNet(nn.Module):
    def __init__(self, in_dim, hidden_size, lr):
        print("  Building UEDNet")
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.encoder_hidden_layer = nn.Linear(
            in_features=self.in_dim, out_features=self.hidden_size
        )
        self.encoder_output_layer = nn.Linear(
            in_features=self.hidden_size, out_features=int(self.hidden_size)
        )

        #self.encoder_decoder_1dconv = nn.Conv1d(
        #    in_channels=int(self.hidden_size/2), kernel_size=3, stride=1, padding=1, out_channels=int(self.hidden_size/2)
        #)
        self.decoder_hidden_layer = nn.Linear(
            in_features=int(self.hidden_size), out_features=self.hidden_size
        )
        self.decoder_output_layer = nn.Linear(
            in_features=self.hidden_size, out_features=self.in_dim
        )

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.criterion = nn.L1Loss()

    def forward(self, features):
        features = features.cuda()
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
    
