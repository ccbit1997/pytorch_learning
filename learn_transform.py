from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2
from PIL import Image

# 1、transform使用Totensor

img_path = "dataset/train/ants/0013035.jpg"
img_PIL = Image.open(img_path)
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img_PIL)

# 2、tensor数据类型
writer = SummaryWriter("logs")
writer.add_image("Tensor_img", img_tensor)
# 3、normalize
print(img_tensor[0, 0, 0])
trans_normal = transforms.Normalize([9, 0, 1], [1, 4, 6])
normal_img = trans_normal(img_tensor)
print(normal_img[0, 0, 0])
writer.add_image("normalize", normal_img, 2)

# 4、resize
print(img_PIL.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img_PIL)
print(img_resize.size)
img_resize_tensor = tensor_trans(img_resize)
writer.add_image("resize", img_resize_tensor)
# 5、compose
trans_resize_2 = transforms.Resize(1080)
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img_PIL)
print(img_resize_2.size())
writer.add_image("resize", img_resize_2, 2)

# 6、Randomcrop
trans_random_crop = transforms.RandomCrop([100, 200])
trans_compose_2 = transforms.Compose([trans_random_crop, tensor_trans])

for i in range(10):
    img_crop = trans_compose_2(img_PIL)
    writer.add_image("random_cropHW", img_crop, i)

writer.close()