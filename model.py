

import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


class ConvBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=16, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=8, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=4, padding='same'),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.blk(x) + x


class DNAEnc(torch.nn.Module):

    def __init__(self, inlen):
        super().__init__()
        self.conv = ConvBlock()
        self.fc0 = nn.Linear(inlen, 64)
        #self.fc1 = nn.Linear(128, 128)
        # self.fc2 = nn.Linear(256, 256)
        self.fc_final = nn.Linear(4, 1)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = x.unsqueeze(1)  # For conv, need channel dim
        x = self.conv(x).squeeze(1)  # Remove the channel dim

        x = F.relu(self.fc0(x.transpose(2, 1)))  # Shape will be 4x128
        #x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = x.transpose(2, 1)
        return F.elu(self.fc_final(x)).squeeze(-1)

