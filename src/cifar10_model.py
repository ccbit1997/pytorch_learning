import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_dataset = torchvision.datasets.CIFAR10(root="../dataset_CIFAR10", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root="../dataset_CIFAR10", train=False, download=True,
                                             transform=torchvision.transforms.ToTensor())

train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)

class CIFAR10_Model(nn.Module):

    def __init__(self):
        super(CIFAR10_Model, self).__init__()
        self.model = Sequential(Conv2d(3, 32, 5, stride=1, padding=2),
                                MaxPool2d(2),
                                Conv2d(32, 32, 5, stride=1, padding=2),
                                MaxPool2d(2),
                                Conv2d(32, 64, 5, stride=1, padding=2),
                                MaxPool2d(2),
                                Flatten(),
                                Linear(1024, 64),
                                Linear(64, 10))

    def forward(self, x):
        output = self.model(x)
        return output

model1 = CIFAR10_Model()
print(model1)

input = torch.ones(64, 3, 32, 32)
print(input.shape)
output = model1(input)
print(output.shape)

writer = SummaryWriter("../logs/model")
writer.add_graph(model1, input)
writer.close()