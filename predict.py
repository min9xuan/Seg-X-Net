import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from net import UNet

def main():
    input_dir = "D:/BaiduNetdiskDownload/all_pics/ring"
    weight_path = "D:/AAA_CodeRepo/models/unet_sclera_iris_pupil_CUDA.pt"
    
    # 为三个类别各自创建输出文件夹
    out_sclera = "D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/sclera"
    out_pupil = "D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/pupil"
    out_iris = "D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/iris"
    out_combined = "D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/combine"
    
    for d in [out_sclera, out_pupil, out_iris, out_combined]:
        os.makedirs(d, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(num_classes=3).to(device)
    if os.path.exists(weight_path):
        net = torch.jit.load(weight_path, map_location=device)
        net.to(device)
    else:
        print("❌ 模型权重加载失败")
        return
        
    net.eval()
    transform = A.Compose([A.Resize(256, 256), A.Normalize((0.5,), (0.5,)), ToTensorV2()])

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    loop = tqdm(image_files, desc="Predicting")

    with torch.no_grad():
        for img_name in loop:
            img_path = os.path.join(input_dir, img_name)
            img_stem = os.path.splitext(img_name)[0]
            
            original_image = cv2.imread(img_path)
            orig_h, orig_w = original_image.shape[:2]
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            augmented = transform(image=image_rgb)
            input_tensor = augmented['image'].unsqueeze(0).to(device)

            logits = net(input_tensor)
            # 获取 3 个通道的预测掩码 (3, 256, 256)
            pred_masks = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy().astype(np.uint8)

            # 后处理保存
            dirs = [out_sclera, out_pupil, out_iris]
            suffixes = ['sclera', 'pupil', 'iris']
            color_mask = np.zeros_like(original_image)

            for c_idx in range(3):
                # 恢复原图分辨率
                mask_resized = cv2.resize(pred_masks[c_idx], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                # 保存各自独立的纯黑白图
                final_mask = mask_resized * 255
                cv2.imwrite(os.path.join(dirs[c_idx], f"{img_stem}_{suffixes[c_idx]}.png"), final_mask)
                
                # 为三通道赋予不同颜色用于合并展示 (BGR格式: 巩膜蓝, 瞳孔绿, 虹膜红)
                if c_idx == 0: color_mask[:, :, 0] = np.maximum(color_mask[:, :, 0], final_mask) # B
                if c_idx == 1: color_mask[:, :, 1] = np.maximum(color_mask[:, :, 1], final_mask) # G
                if c_idx == 2: color_mask[:, :, 2] = np.maximum(color_mask[:, :, 2], final_mask) # R

            # 叠加到原图上保存为彩色预览图
            combined_vis = cv2.addWeighted(original_image, 0.7, color_mask, 0.3, 0)
            cv2.imwrite(os.path.join(out_combined, f"{img_stem}_combined.png"), combined_vis)

if __name__ == '__main__':
    main()