import os
from pathlib import Path

def rename_sclera_masks(folder_path):
    """
    移除文件夹内图片文件名中的 '_sclera' 后缀。
    例如: 000005_sclera.png -> 000005.png
    """
    folder = Path(folder_path)
    
    # 检查文件夹是否存在
    if not folder.exists() or not folder.is_dir():
        print("指定的文件夹不存在或不是一个目录，请检查路径。")
        return

    count = 0
    # 遍历文件夹中的所有文件
    for file_path in folder.iterdir():
        # 确保是文件，并且文件名中包含 '_sclera'
        if file_path.is_file() and '_sclera' in file_path.name:
            # 构造新的文件名，将 '_sclera' 替换为空字符串
            new_name = file_path.name.replace('_sclera', '')
            new_file_path = folder / new_name
            
            # 安全检查：如果目标文件名已经存在，给出警告并跳过，防止覆盖
            if new_file_path.exists():
                print(f"⚠️ 警告: {new_name} 已存在，跳过重命名 {file_path.name}")
                continue
            
            # 执行重命名
            file_path.rename(new_file_path)
            count += 1
            
    print(f"✅ 重命名完成！共成功修改了 {count} 个文件。")

if __name__ == '__main__':
    # ================= 使用说明 =================
    # 请将这里的路径替换为你存放转换后掩码的文件夹路径
    # Windows 用户建议使用 r"路径" 的格式，例如 r"D:\dataset\masks"
    
    TARGET_FOLDER = r"D:\\AAA_DataRepo\\Tsinghua_muzhen_data\\labels"
    
    rename_sclera_masks(TARGET_FOLDER)