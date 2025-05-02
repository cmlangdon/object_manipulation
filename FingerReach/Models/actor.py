import torch.nn as nn
import torch

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.STATE_DIM = 115
        self.ACTION_DIM = 39
        self.mlp = nn.Sequential(nn.Linear(self.STATE_DIM, 8, bias=False),
                                 nn.ReLU6(),
                                 nn.Linear(8, 16, bias=False),
                                 nn.ReLU6(),
                                 nn.Linear(16, 32, bias=False),
                                 nn.ReLU6(),
                                 nn.Linear(32, 64, bias=False),
                                 nn.ReLU6(),
                                 nn.Linear(64, 32, bias=False),
                                 nn.ReLU6(),
                                 nn.Linear(32, 16, bias=False),
                                 nn.ReLU6(),
                                 nn.Linear(16, 8, bias=False),
                                 nn.ReLU6(),
                                 nn.Linear(8, self.ACTION_DIM, bias=False))

    def forward(self, x):
        return self.mlp(x)