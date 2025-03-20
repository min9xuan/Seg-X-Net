import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from net import UNet

model = UNet(num_classes=4)
# 将模型设置为评估模式
model.eval()

# 读取图像，使用OpenCV以RGB模式读取图片
img = cv2.imread("negative_403.jpg", 1)
print("img: ", img)

# 将读取的图像从numpy数组转换为PIL格式的RGB图像
img = Image.fromarray(img, mode="RGB")

# 定义图像的预处理步骤，使用torchvision.transforms进行处理
trans = transforms.Compose([
    transforms.Resize([640, 960]),  # 调整图像大小
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 进行标准化处理
])

# 对图像进行预处理
img = trans(img)
print("Processed img: ", img)

# 保存预处理后的图像（可以用于调试或可视化）
save_image(img, "TestImg.jpg")

# 为了进行批量处理，扩展图像的维度，使其成为一个batch的形式 [batch_size, channels, height, width]
img = img.unsqueeze(0)  
print("After unsqueeze, img: ", img)

# 将图像移动到相应的设备（比如GPU）
img = img.to(device)

# 可选择加载已序列化的模型（在此注释掉）
# model2 = torch.jit.load("model.pt")
# output = model2(img)
# print("output:", output)

# 使用torch.jit.trace对模型进行跟踪，生成TorchScript模型
traced_net = torch.jit.trace(model, img)
print("traced_net: ", traced_net)

# 将traced模型保存为文件
traced_net.save("model.pt")

print("模型序列化导出成功")
