import cv2
import os
from pathlib import Path

def convert_masks_to_binary(input_dir, output_dir):
    """
    将文件夹内的所有彩色掩码转换为黑底白前景的二值掩码。
    """
    # 确保输出文件夹存在，如果不存在则创建
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取支持的图像文件列表 (这里以 .png 和 .jpg 为例)
    input_path = Path(input_dir)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_path.glob(ext))

    if not image_paths:
        print("未在输入文件夹中找到图像文件，请检查路径。")
        return

    count = 0
    for img_path in image_paths:
        # 1. 以灰度模式读取图像
        # 原图是彩色的，转为灰度后，背景仍为0，彩色前景会变成大于0的灰度值
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"警告: 无法读取图像 {img_path.name}")
            continue

        # 2. 二值化处理：将所有大于 0 的像素点设为 255 (纯白)
        _, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

        # 3. 保存到输出文件夹
        output_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(output_path), binary_mask)
        count += 1

    print(f"处理完成！共成功转换 {count} 张掩码图像。")
    print(f"文件已保存至: {output_dir}")

if __name__ == '__main__':
    # ================= 使用说明 =================
    # 请将下面的路径替换为你实际的文件夹路径
    # 注意：Windows 路径建议使用双斜杠 \\ 或在字符串前加 r，例如 r"D:\dataset\masks"
    
    INPUT_FOLDER = r"D:\\AAA_OCT\\data_annotation\\Tsinghua_muzhen_data\\target\\SegmentationClass"   # 原始彩色掩码所在文件夹
    OUTPUT_FOLDER = r"D:\\AAA_DataRepo\\Tsinghua_muzhen_data\\labels" # 转换后黑底白图保存的文件夹
    
    convert_masks_to_binary(INPUT_FOLDER, OUTPUT_FOLDER)