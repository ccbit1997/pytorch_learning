import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, Softmax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_dataset = torchvision.datasets.CIFAR10(root="../dataset_CIFAR10", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root="../dataset_CIFAR10", train=False, download=True,
                                             transform=torchvision.transforms.ToTensor())

test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, drop_last=True)

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
                                Linear(64, 10),
                                Softmax(1))

    def forward(self, x):
        output = self.model(x)
        return output


model1 = CIFAR10_Model()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)

for epoch in range(20):
    running_loss = 0.0
    for data in test_dataloader:
        img, gt_label = data
        output = model1(img)
        ce_loss = loss(output, gt_label)
        optim.zero_grad()
        ce_loss.backward()
        optim.step()
        running_loss += ce_loss
    print(running_loss)

