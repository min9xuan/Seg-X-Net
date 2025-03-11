import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from net import UNet

# 定义测试集数据加载
class TestDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        # 获取所有图片文件（支持jpg, jpeg, png格式）
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 获取图片路径
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 加载标签并合并
        base_name = os.path.splitext(image_file)[0]
        iris_path = os.path.join(self.label_dir, f'{base_name}_iris.png')
        pupil_path = os.path.join(self.label_dir, f'{base_name}_pupil.png')
        sclera_path = os.path.join(self.label_dir, f'{base_name}_sclera.png')

        # 加载标签：若标签不存在，填充为全0
        iris_label = np.array(Image.open(iris_path)) if os.path.exists(iris_path) else np.zeros((512, 512), dtype=np.uint8)
        pupil_label = np.array(Image.open(pupil_path)) if os.path.exists(pupil_path) else np.zeros((512, 512), dtype=np.uint8)
        sclera_label = np.array(Image.open(sclera_path)) if os.path.exists(sclera_path) else np.zeros((512, 512), dtype=np.uint8)

        # 合并标签：0=背景，1=虹膜，2=瞳孔，3=巩膜
        label = np.zeros_like(iris_label, dtype=np.uint8)
        label[iris_label > 0] = 1
        label[pupil_label > 0] = 2
        label[sclera_label > 0] = 3

        # 图片预处理
        transform_image = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),  # 使用双线性插值
            transforms.ToTensor()
        ])

        transform_label = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),  # 使用双线性插值
            transforms.ToTensor()
        ])

        image = transform_image(image)
        label = transform_label(Image.fromarray(label)).squeeze(0).long()

        return image, label

# 定义加载模型函数
def load_model(weight_path, num_classes, device):
    net = UNet(num_classes).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print('Model loaded successfully.')
    else:
        raise FileNotFoundError(f"Model weights not found at {weight_path}. Please check the path.")
    return net


def evaluate_metrics(output, label, num_classes):
    sd_per_class = []
    iou_per_class = []
    pa_per_class = []  # 存储每个类别的像素准确率

    for class_idx in range(num_classes):
        # 获取当前类别的预测和标签位置
        class_output = output == class_idx
        class_label = label == class_idx

        # 计算标准差
        class_output_flat = class_output.astype(np.float32).flatten()
        class_label_flat = class_label.astype(np.float32).flatten()
        sd = np.std(class_output_flat - class_label_flat)
        sd_per_class.append(sd)

        # 计算 IoU
        intersection = np.sum(class_output_flat * class_label_flat)
        union = np.sum(class_output_flat) + np.sum(class_label_flat) - intersection
        iou = intersection / (union + 1e-6)  # 避免除零错误
        iou_per_class.append(iou)

        # 计算 Pixel Accuracy
        correct_pixels = np.sum(class_output_flat == class_label_flat)
        total_pixels = class_output_flat.size
        pa = correct_pixels / total_pixels
        pa_per_class.append(pa)

    # 计算 Mean_SD、mIoU 和 MPA
    mean_sd = np.mean(sd_per_class)
    mIoU = np.mean(iou_per_class)
    MPA = np.mean(pa_per_class)  # 计算均值像素准确率

    return mean_sd, mIoU, MPA



# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = 'params/unet.pth'
    image_dir = 'data/test/image'
    label_dir = 'data/test/labels'
    result_path = 'test_result'

    # 定义测试集和网络
    num_classes = 4  # 包括背景
    test_dataset = TestDataset(image_dir, label_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    net = load_model(weight_path, num_classes, device)

    # 评估模型
    net.eval()
    os.makedirs(result_path, exist_ok=True)

    mean_sds = []
    mean_ious = []
    mean_pas = []  # 存储每张图像的像素准确率

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)

            # 模型推理
            output = net(image)
            output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            label = label.squeeze(0).cpu().numpy()

            # 计算评估指标
            mean_sd, mIoU, MPA = evaluate_metrics(output, label, num_classes)
            mean_sds.append(mean_sd)
            mean_ious.append(mIoU)
            mean_pas.append(MPA)

            # 可视化并保存结果
            output_colors = {
                0: [0, 0, 0],  # 背景
                1: [255, 0, 0],  # 虹膜
                2: [0, 255, 0],  # 瞳孔
                3: [0, 0, 255]  # 巩膜
            }

            result_img = np.zeros((*output.shape, 3), dtype=np.uint8)
            for class_idx, color in output_colors.items():
                result_img[output == class_idx] = color

            # 使用原始图像文件名作为结果文件名
            original_image_name = test_dataset.image_files[i]
            result_file = os.path.join(result_path, original_image_name)  # 保留原文件名
            result_img_resized = cv2.resize(result_img,
                                            (cv2.imread(os.path.join(image_dir, original_image_name)).shape[1],
                                             cv2.imread(os.path.join(image_dir, original_image_name)).shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(result_file, cv2.cvtColor(result_img_resized, cv2.COLOR_RGB2BGR))


    # 输出评估结果
    print("\n=== Performance Metrics ===")
    print(f"Mean SD: {np.mean(mean_sds):.4f}")
    print(f"Mean IoU: {np.mean(mean_ious):.4f}")
    print(f"Mean Pixel Accuracy (MPA): {np.mean(mean_pas):.4f}")  # 输出 MPA




if __name__ == '__main__':
    main()