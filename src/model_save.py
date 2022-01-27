import torch
import torchvision

"""
方式一：模型结构+模型参数
"""
vgg16 = torchvision.models.vgg16(pretrained=False)

torch.save(vgg16, "../model/vgg16_method1.pth")

"""
方式二：模型参数（官方推荐）
"""

torch.save(vgg16.state_dict(), "../model/vgg16_method2.pth")