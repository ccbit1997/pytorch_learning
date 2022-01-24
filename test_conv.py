import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]], dtype=torch.float32)

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]], dtype=torch.float32)
dataset = torchvision.datasets.CIFAR10("./dataset_CIFAR10", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)

class maxpool_model(nn.Module):
    def __init__(self):
        super(maxpool_model, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool(x)
        return output

# input = torch.reshape(input, (-1, 1, 5, 5))
# kernel = torch.reshape(kernel, (1, 1, 3, 3))
# output = F.conv2d(input=input, weight=kernel, stride=1, padding=1)
# output2 = F.conv2d(input, kernel, stride=2)
maxpool1 = maxpool_model()
# output3 = maxpool1(input)

writer = SummaryWriter("logs/maxpool")
step = 0
for data in dataloader:
    img, label = data
    writer.add_images("raw_image", img, global_step=step)
    maxpool_output = maxpool1(img)
    writer.add_images("maxpool", maxpool_output, global_step=step)
    step += 1

writer.close()
