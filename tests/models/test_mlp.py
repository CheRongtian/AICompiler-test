import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def export_case():
    torch.manual_seed(0)

    model = MLPModel().eval()
    x = torch.randn(1, 8, 64)

    return model, (x,)