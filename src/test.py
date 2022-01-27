import os
# import cv2
import argparse

import torch
import torchvision.transforms
from PIL import Image
from torch import nn

image_path = "../img/airplane.png"
# img = cv2.imread(image_path)
img = Image.open(image_path)
img = img.convert("RGB")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
img_tensor = transform(img)
print(img_tensor.shape)

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


device = torch.device("cuda:0")
model_param = torch.load("../model/cifar10_model1.pth")
model = CIFAR10_model()
model.load_state_dict(model_param)
model.to(device)
print(model)
img_tensor = torch.reshape(img_tensor, (1, 3, 32, 32))
img_tensor = img_tensor.to(device)

model.eval()
with torch.no_grad():
    output = model(img_tensor)

print(output)
print(output.argmax(1))

