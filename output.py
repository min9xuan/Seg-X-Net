import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from net import UNet

def load_model(weight_path, num_classes, device):
    try:
        net = UNet(num_classes).to(device)
        state_dict = torch.load(weight_path, map_location=device)
        net.load_state_dict(state_dict)
        print('Model loaded successfully.')
    except TypeError:
        net = torch.jit.load(weight_path, map_location=device)
        print('TorchScript model loaded successfully.')
    return net

# 处理单张图像
def process_single_image(image_path, label_dir, weight_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 4

    net = load_model(weight_path, num_classes, device)
    net.eval()

    image = Image.open(image_path).convert('RGB')
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    iris_path = os.path.join(label_dir, f'{base_name}_iris.png')
    pupil_path = os.path.join(label_dir, f'{base_name}_pupil.png')
    sclera_path = os.path.join(label_dir, f'{base_name}_sclera.png')

    transform_image = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    image_tensor = transform_image(image).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output = net(image_tensor)
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 可视化输出
    output_colors = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
    result_img = np.zeros((*output.shape, 3), dtype=np.uint8)
    for class_idx, color in output_colors.items():
        result_img[output == class_idx] = color

    # 调整回原图大小
    original_size = cv2.imread(image_path).shape[:2][::-1]
    result_img_resized = cv2.resize(result_img, original_size, interpolation=cv2.INTER_LINEAR)

    os.makedirs(output_path, exist_ok=True)
    result_file = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(result_file, cv2.cvtColor(result_img_resized, cv2.COLOR_RGB2BGR))
    print(f"Result saved to {result_file}")

if __name__ == '__main__':
    image_path = 'data/test/image/001.jpg'
    label_dir = 'data/test/labels'
    weight_path = 'params/unet.pth'
    output_path = 'result/single_result'

    process_single_image(image_path, label_dir, weight_path, output_path)
