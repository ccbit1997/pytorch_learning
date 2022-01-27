import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from src.model import CIFAR10_model

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset_CIFAR10", transform=torchvision.transforms.ToTensor(),
                                          train=True, download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset_CIFAR10", transform=torchvision.transforms.ToTensor(),
                                         train=False, download=True)
# 数据集大小
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集长度为: {}".format(train_data_size))
print("测试数据集长度为: {}".format(test_data_size))

# 利用DataLoader来加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
model1 = CIFAR10_model()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
learining_rate = 1e-2
optim = torch.optim.SGD(model1.parameters(), lr=learining_rate, momentum=0.9)

# 设置网络的一些参数
# 记录训练次数
total_train_step = 0
total_test_step = 0

# 训练轮数
epoch = 20

# 添加Tensorboard
writer = SummaryWriter("../logs/loss")

for i in range(epoch):

    start_time = time.time()
    model1.train()
    print("-------第 {} 轮训练开始--------".format(i+1))
    for data in train_dataloader:
        img, label = data
        output = model1(img)
        cur_loss = loss_fn(output, label)
        # 优化器调优
        optim.zero_grad()
        cur_loss.backward()
        optim.step()

        total_train_step += 1

        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            start_time = end_time
            print("训练次数: {}，loss = {}".format(total_train_step, cur_loss.item()))
            writer.add_scalar("train_loss", cur_loss.item(), global_step= total_train_step)


    # 测试步骤开始
    model1.eval()
    total_test_loss = 0.0
    right_data = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, label = data
            outputs = model1(img)
            right_data += (outputs.argmax(1) == label).sum()
            test_loss = loss_fn(outputs, label)
            total_test_loss += test_loss.item()
        print("整体测试集Loss: {}".format(total_test_loss))
        print("正确率: {}".format(right_data/test_data_size))
        writer.add_scalar("test_Loss", total_test_loss, global_step=total_test_step)
        writer.add_scalar("test_accuracy", right_data/test_data_size, global_step=total_test_step)
        total_test_step += 1

    torch.save(model1, "../model/model_{}.pth".format(i))
    print("模型已保存")

writer.close()

torch.save(model1.state_dict(), "../model/cifar10_model1.pth")

