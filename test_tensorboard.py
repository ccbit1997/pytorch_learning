from torch.utils.tensorboard import SummaryWriter
import cv2

writer = SummaryWriter("logs")
img_path = "dataset/train/bees/16838648_415acd9e3f.jpg"
img = cv2.imread(img_path)
print(type(img))
print(img.shape)
writer.add_image("test", img, 2, dataformats="HWC")
for i in range(50):
    writer.add_scalar("y=3x", 3*i, i)

writer.close()