import os
import cv2
import numpy as np
import torch
from sympy import Number
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from net import UNet
from evaluate import *

# 定义测试集数据加载
class TestDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')

        base_name = os.path.splitext(image_file)[0]
        iris_path = os.path.join(self.label_dir, f'{base_name}_iris.png')
        pupil_path = os.path.join(self.label_dir, f'{base_name}_pupil.png')
        sclera_path = os.path.join(self.label_dir, f'{base_name}_sclera.png')

        iris_label = np.array(Image.open(iris_path)) if os.path.exists(iris_path) else np.zeros((512, 512), dtype=np.uint8)
        pupil_label = np.array(Image.open(pupil_path)) if os.path.exists(pupil_path) else np.zeros((512, 512), dtype=np.uint8)
        sclera_label = np.array(Image.open(sclera_path)) if os.path.exists(sclera_path) else np.zeros((512, 512), dtype=np.uint8)

        label = np.zeros_like(iris_label, dtype=np.uint8)
        label[iris_label == 255] = 1
        label[pupil_label == 255] = 2
        label[sclera_label == 255] = 3

        transform_image = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        resize = transforms.Resize((512, 512))

        image = transform_image(image)
        label_pil = Image.fromarray(label.astype(np.uint8))
        label_resized = resize(label_pil)
        label = np.array(label_resized)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

# 载入模型
def load_model(weight_path, num_classes, device):
    net = UNet(num_classes).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print('Model loaded successfully.')
    else:
        raise FileNotFoundError(f"Model weights not found at {weight_path}.")
    return net

# 评估模型
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = 'params/unet.pth'
    image_dir = 'data/test/image'
    label_dir = 'data/test/labels'
    result_path = 'result'
    os.makedirs(result_path, exist_ok=True)

    num_classes = 4
    test_dataset = TestDataset(image_dir, label_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    net = load_model(weight_path, num_classes, device)
    net.eval()

    results = []

    dice_sum = 0
    iou_sum = 0
    sensitivity_sum = 0
    accuracy_sum = 0
    kappa_sum = 0
    specificity_sum = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            output = net(image)
            output = torch.argmax(output, dim=1).squeeze(0).cpu()
            label = label.squeeze(0).cpu()

            # 计算评估指标
            dice = diceCoeff(output, label)
            iou = mean_iou(output, label)
            sensitivity = compute_sensitivity(output, label)
            accuracy = compute_acc(output, label)
            kappa = compute_kappa(output, label)
            specificity = compute_specificity(output, label)

            dice_sum += dice
            iou_sum += iou
            sensitivity_sum += sensitivity
            accuracy_sum += accuracy
            kappa_sum += kappa
            specificity_sum += specificity

            results.append(f"Image {i}: Dice={dice:.4f}, IoU={iou:.4f}, Sensitivity={sensitivity:.4f}, Accuracy={accuracy:.4f}, Kappa={kappa:.4f}, Specificity={specificity:.4f}")

            # 检查 IoU 和 Kappa 是否为负值
            if iou < 0 or kappa < 0:
                print(f"Warning: Negative IoU or Kappa for Image {i}. IoU: {iou}, Kappa: {kappa}")
                print(f"Output for Image {i}: {output}")
                print(f"Label for Image {i}: {label}")

            # 生成结果图像
            output_colors = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
            result_img = np.zeros((*output.shape, 3), dtype=np.uint8)
            for class_idx, color in output_colors.items():
                result_img[output == class_idx] = color

            # 保存结果图像
            original_image_name = test_dataset.image_files[i]
            result_file = os.path.join(result_path, original_image_name)
            result_img_resized = cv2.resize(result_img,
                                            (cv2.imread(os.path.join(image_dir, original_image_name)).shape[1],
                                             cv2.imread(os.path.join(image_dir, original_image_name)).shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(result_file, cv2.cvtColor(result_img_resized, cv2.COLOR_RGB2BGR))

    # 保存评估结果到文件
    with open(os.path.join(result_path, 'evaluation_results.txt'), 'w') as f:
        f.write('\n'.join(results))

    print("Evaluation completed. Results saved in result/evaluation_results.txt")
    print("m_dice:", (dice_sum / len(test_loader)).numpy())
    print("m_iou:", (iou_sum / len(test_loader)).numpy())
    print("m_sensitivity:", sensitivity_sum / len(test_loader))
    print("m_accuracy:", accuracy_sum / len(test_loader))
    print("m_kappa:", (kappa_sum / len(test_loader)).numpy())
    print("m_specificity:", (specificity_sum / len(test_loader)).numpy())

if __name__ == '__main__':
    main()
