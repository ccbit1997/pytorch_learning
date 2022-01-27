import torch
from torch import nn


class CIFAR10_model(nn.Module):

    def __init__(self):
        super(CIFAR10_model, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 32, 5, stride=1, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(32, 32, 5, stride=1, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(32, 64, 5, stride=1, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.Flatten(),
                                   nn.Linear(1024, 64),
                                   nn.Linear(64, 10))

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":

    model1 = CIFAR10_model()
    input_tensor = torch.ones((64, 3, 32, 32))
    output = model1(input_tensor)
    print(output.shape)