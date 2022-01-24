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
train_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=0)


# print(test_dataset[0])
# print(test_dataset.classes)
# img, label = test_dataset[0]
# print(img)
# print(label)
# print(test_dataset.classes[label])
# img.show()
# print(test_dataset[0])
writer = SummaryWriter("../logs")
classes = test_dataset.classes
for i in range(10):
    img_tensor, label = test_dataset[i]
    print(classes[label])
    writer.add_image(classes[label], img_tensor, i)

writer.close()
