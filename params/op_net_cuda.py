import torch
from torch import nn
from torch.nn import functional as F

# 假设你有一个UNet模型定义（根据你的实际模型来调整）
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
        # forword 的作用是设定x的数据流向
    def forward(self,x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)  # 跳跃连接，将 out 和 feature_map 进行特征融合


class UNet(nn.Module):
    def __init__(self,num_classes):
        super(UNet, self).__init__()
        self.c1=Conv_Block(3,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out=nn.Conv2d(64,num_classes,3,1,1)

    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1=self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载模型定义并转移到GPU
model = UNet(2).to(device)

# 加载权重（自动映射到当前设备）
model_path = "D:/AAA_git_myrepo/models/unet_seg_vessels.pth"
model.load_state_dict(torch.load(model_path, map_location=device))  # 关键修改：添加map_location参数
model.eval()  # 设置为评估模式

# 创建示例输入并转移到GPU
example_input = torch.randn(1, 3, 512, 512).to(device)  # 关键修改：添加.to(device)

# 使用TorchScript进行转化（现在全部在GPU上运行）
with torch.no_grad():  # 添加无梯度上下文以提升效率
    traced_model = torch.jit.trace(model, example_input)

# 保存为.pt文件
output_path = "D:/AAA_git_myrepo/models/unet_seg_vessels_CUDA.pt"
traced_model.save(output_path)

print(f"模型已成功保存为 {output_path}")
