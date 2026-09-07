import torch
import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        y = self.linear(x)
        return self.norm(y + x)


def export_case():
    torch.manual_seed(0)

    model = TestModel().eval()
    x = torch.randn(1, 8, 64)

    return model, (x,)