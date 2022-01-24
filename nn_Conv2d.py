import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(
    root="./dataset_CIFAR10",
    train=True,
    download=True,
    transform=dataset_transform)
test_dataset = torchvision.datasets.CIFAR10(
    root = "./dataset_CIFAR10",
    train=False,
    download=True,
    transform=dataset_transform
)

dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

model1 = my_model()
writer = SummaryWriter("logs/convs")
step = 0
for data in dataloader:
    img, label = data
    writer.add_images("raw_images", img, global_step=step)
    # torch.Size([64, 3, 32, 32])
    # torch.Size([64, 6, 30, 30])
    output = model1(img)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("conv_images", output, global_step=step)
    step += 1
    print(img.shape)
    print(output.shape)

writer.close()
