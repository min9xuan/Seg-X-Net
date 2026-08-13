import os

def rename_images(folder_path):
    # 获取所有 .jpg 文件并按文件名排序
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    files.sort()  # 确保按文件名排序

    # 遍历文件并重命名
    for index, file_name in enumerate(files, start=1):
        # 构造新的文件名，确保为六位数递增
        new_name = f"{index:06d}.jpg"
        # 获取旧文件路径和新文件路径
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

# 指定JPEGImages文件夹路径
jpeg_images_folder = "JPEGImages"
rename_images(jpeg_images_folder)
