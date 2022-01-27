import torchvision
from torch import nn
# train_dataset = torchvision.datasets.ImageNet(root="../dataset_ImageNet", transform=torchvision.transforms.ToTensor(),
#                                               split='train', download=True)

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
print(vgg16_pretrain)

train_dataset = torchvision.datasets.CIFAR10(root="../dataset_CIFAR10", transform=torchvision.transforms.ToTensor(),
                                             train=True, download=True)

vgg16_pretrain.classifier.add_module('add_linear', nn.Linear(1000, 10)) #模型修改-增加层
vgg16.classifier[6] = nn.Linear(4096, 10) #模型修改-修改层
print(vgg16)