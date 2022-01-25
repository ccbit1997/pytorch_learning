import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="../dataset_CIFAR10", transform=torchvision.transforms.ToTensor(),
                                       train=False, download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64, drop_last=True)

class model_cc(nn.Module):

    def __init__(self):
        super(model_cc, self).__init__()
        self.linear1 = nn.Linear(in_features=196608, out_features=10)

    def forward(self, x):
        output = self.linear1(x)
        return output


model1 = model_cc()
for data in dataloader:
    img, label = data
    print(img.shape)
    input = torch.flatten(img)
    print(input.shape)
    output = model1(input)
    print(output.shape)