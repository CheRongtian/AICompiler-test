import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 32)

    def forward(self, x):
        return self.linear(x)


def export_case():
    torch.manual_seed(0)

    model = LinearModel().eval()
    x = torch.randn(1, 8, 64)

    return model, (x,)