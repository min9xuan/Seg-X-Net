import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        """
        血管分割数据集
        
        Args:
            root_dir (str): 根目录路径，包含110-169文件夹和mask文件夹
            transform: 应用于原图像的变换
            mask_transform: 应用于掩码的变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # 获取所有图像文件路径
        self.image_paths = []
        self.mask_paths = []
        
        # 遍历根目录下的所有子文件夹（排除mask文件夹）
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            
            # 跳过mask文件夹和非目录文件
            if not os.path.isdir(item_path) or item == 'mask':
                continue
            
            folder_name = item  # 文件夹名称
            
            # 获取该文件夹下的所有图片文件
            image_files = glob.glob(os.path.join(item_path, "*.jpg")) + \
                         glob.glob(os.path.join(item_path, "*.png")) + \
                         glob.glob(os.path.join(item_path, "*.jpeg"))
            
            for img_path in image_files:
                # 获取文件名（不含扩展名）
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # 构造对应的掩码文件名：文件夹名 + 图片名
                mask_name = f"{folder_name}{img_name}"
                
                # 尝试多种掩码文件扩展名
                mask_found = False
                for ext in ['.jpg', '.png', '.jpeg']:
                    mask_path = os.path.join(root_dir, "mask", mask_name + ext)
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                        mask_found = True
                        break
                
                if not mask_found:
                    print(f"警告: 未找到对应的掩码文件: {mask_name}")
        
        print(f"找到 {len(self.image_paths)} 对图像-掩码文件")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 读取掩码
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')  # 转为灰度图
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 应用掩码变换
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # 如果mask还不是tensor，需要转换
        if not isinstance(mask, torch.Tensor):
            mask = transforms.ToTensor()(mask)
            
        # 将掩码转换为二值掩码（0和1）用于CrossEntropyLoss
        # 假设掩码中白色区域（血管）为前景类别1，黑色为背景类别0
        mask = (mask > 0.5).long().squeeze(0)  # 转换为long类型并去掉channel维度
        
        return image, mask
    
    def get_sample_info(self, idx):
        """获取样本信息，用于调试"""
        return {
            'image_path': self.image_paths[idx],
            'mask_path': self.mask_paths[idx]
        }


# 使用示例
if __name__ == "__main__":
    # 创建数据集实例
    dataset = MyDataset(root_dir="your_root_directory")
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # 测试读取一个batch
    for batch in dataloader:
        images = batch['image']
        masks = batch['mask']
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        break
    
    # 查看前几个样本的信息
    print("\n前5个样本的路径信息:")
    for i in range(min(5, len(dataset))):
        info = dataset.get_sample_info(i)
        print(f"样本 {i}: {info['image_path']} -> {info['mask_path']}")