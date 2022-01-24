import torch
from torch import nn


class model_cc(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        output = 2*x
        return output

model_1 = model_cc()
x = torch.tensor(1.0)
z = 2
y = model_1.forward(z)
print(y)