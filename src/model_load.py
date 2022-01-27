import torch
import torchvision

"""
方式一：对应 torch.save(vgg16, "../model/vgg16_method1.pth")
陷阱：自定义模型需引入所定义的类
"""

# vgg16 = torch.load("../model/vgg16_method1.pth")
# print(vgg16)

"""
方式2：加载模型
"""
model_load = torch.load("../model/vgg16_method2.pth")
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(model_load)
print(vgg16)