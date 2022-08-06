import torch
import torch.nn as nn


class MConstrastEncoder(nn.Module):

    def __init__(self, n_c):
        super(MConstrastEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(n_c, 3, (3, 3), 2),
            nn.LeakyReLU(),
            nn.Conv2d(3, 6, (3, 3), 2),
            nn.LeakyReLU(),
            nn.Conv2d(6, 9, (3, 3), 2),
            nn.LeakyReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.network(x)
        

class MConstrastProjector(nn.Module):

    def __init__(self, input, output):
        super(MConstrastProjector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input, output)
        )

    def forward(self, x):
        return nn.functional.normalize(self.network(x))