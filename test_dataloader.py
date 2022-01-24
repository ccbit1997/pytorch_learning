import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_dataset = torchvision.datasets.CIFAR10(
    root="./dataset_CIFAR10",
    train=True,
    download=True,
    transform=dataset_transform)
test_dataset = torchvision.datasets.CIFAR10(
    root="./dataset_CIFAR10",
    train=False,
    download=True,
    transform=dataset_transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
img0, label0 = test_dataset[0]
print(img0.shape)
print(label0)

writer = SummaryWriter("dataloader")

for epoch in range(3):
    step = 0
    for data in test_dataloader:
        img, label = data
        writer.add_images("Epoch: {}".format(epoch), img, step)
        step += 1
        # print(img.shape)
        # print(label)
writer.close()