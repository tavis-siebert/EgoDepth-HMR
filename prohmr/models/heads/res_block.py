
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.h = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        return x + self.fc2(self.h(self.fc1(x)))