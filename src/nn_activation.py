import torch
from torch import nn
from torch.nn import ReLU
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)
dataset = torchvision.datasets.CIFAR10(root="../dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64)

class model_cc(nn.Module):
    def __init__(self):
        super(model_cc, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = nn.Sigmoid()
    def forward(self, x):
        output = self.sigmoid1(x)
        return output

model1 = model_cc()
# output = model1(input)
# print(output)
writer = SummaryWriter("../logs/relu")
step = 0

for data in dataloader:
    img, label = data
    writer.add_images("raw_images", img, global_step=step)
    output = model1(img)
    writer.add_images("sigmoid_images", output, global_step=step)
    step += 1

writer.close()