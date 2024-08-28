import torch.nn as nn

class DropoutScaling(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.scale = 1.0 / (1.0 - p)

    def forward(self, x):
        return x * self.scale