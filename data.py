import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# 定义图像和标签的变换，调整尺寸为256x256并转换为张量
transform_image = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小为512x512
    transforms.ToTensor(), # 转换为张量
])

transform_label = transforms.Compose([
    transforms.Resize((512, 512), Image.NEAREST)  # 调整分割标签大小，使用最近邻插值
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.image_names = os.listdir(os.path.join(path, 'JPEGImages'))  # 原始图片名列表

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # 获取当前图片名
        image_name = self.image_names[index]  
        base_name = os.path.splitext(image_name)[0] 

        # 构建路径
        image_path = os.path.join(self.path, 'JPEGImages', image_name)
        iris_path = os.path.join(self.path, 'SegmentationClass', f"{base_name}_iris.png")
        pupil_path = os.path.join(self.path, 'SegmentationClass', f"{base_name}_pupil.png")
        sclera_path = os.path.join(self.path, 'SegmentationClass', f"{base_name}_sclera.png")

        # 读取原始图像
        image = Image.open(image_path).convert("RGB")  # 转为 RGB 格式

        # 读取分割标签
        iris = Image.open(iris_path).convert("L")  # 单通道灰度图
        pupil = Image.open(pupil_path).convert("L")
        sclera = Image.open(sclera_path).convert("L")

        # 将不同分割部分合并为单一标签图
        segment_image = np.zeros_like(np.array(iris))  # 初始化标签图
        segment_image[np.array(iris) > 0] = 1  # 虹膜标签设为 1
        segment_image[np.array(pupil) > 0] = 2  # 瞳孔标签设为 2
        segment_image[np.array(sclera) > 0] = 3  # 巩膜标签设为 3

        # 调整尺寸
        image = transform_image(image)  # 调整图像大小并转换为张量
        segment_image = transform_label(Image.fromarray(segment_image))  # 调整标签大小

        return image, torch.tensor(np.array(segment_image), dtype=torch.long)


if __name__ == '__main__':
    dataset = MyDataset('data')
    print(f"Dataset size: {len(dataset)}")

    # 测试第一个样本
    image, segment_image = dataset[0]
    print(f"Image shape: {image.shape}")  # 图像张量的形状 [C, 256, 256]
    print(f"Segment shape: {segment_image.shape}")  # 分割标签的形状 [256, 256]
    print(f"Unique labels in segment: {torch.unique(segment_image)}")  # [0, 1, 2, 3]
