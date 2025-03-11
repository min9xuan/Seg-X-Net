import os

import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'
if __name__ == '__main__':
    num_classes = 3 + 1  # +1是背景也为一类
    data_loader = DataLoader(MyDataset(data_path), batch_size=2, shuffle=True)
    net = UNet(num_classes).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()

    epoch = 1
    while epoch < 200:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device) # 设备选择
            # 前向传递
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image.long())
            # 反向传播
            opt.zero_grad()
            train_loss.backward()

            # 更新参数
            opt.step()

            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
                # 原始图像
                _image = image[0]  # (C, H, W)
                # 分割标签（归一化）
                _segment_image = segment_image[0].unsqueeze(0).float() / 3.0  # (1, H, W)
                # 预测结果（归一化）
                _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0).float() / 3.0  # (1, H, W)
                # 拼接图像: 将3个单通道图像按通道堆叠为1个RGB图像
                img = torch.cat([_segment_image, _out_image, _image[:1]], dim=0)  # (3, H, W)
                # 确保路径存在
                os.makedirs(save_path, exist_ok=True)
                # 保存图像
                save_image(img, f'{save_path}/{i}.png')
        if epoch % 20 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully!')
        epoch += 1
