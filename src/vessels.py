import os
import argparse
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from net import *
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

# 定义图像和标签的变换，调整尺寸为256x256并转换为张量
transform_image = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小为512x512
    transforms.ToTensor(), # 转换为张量
])

transform_label = transforms.Compose([
    transforms.Resize((512, 512), Image.NEAREST)  # 调整分割标签大小，使用最近邻插值
])

def print_color(text, color="green"):
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def show_gpu_info():
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem_allocated = torch.cuda.memory_allocated(idx) / 1024**2  # MB
        mem_total = torch.cuda.get_device_properties(idx).total_memory / 1024**2  # MB
        print_color(f"🖥️  GPU: {name} | Allocated: {mem_allocated:.1f}MB / {mem_total:.1f}MB", "yellow")
    else:
        print_color("🖥️  Running on CPU", "yellow")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_color(f"Using device: {device}", "yellow")

    data_loader = DataLoader(MyDataset(args.data_path, transform_image, transform_label), batch_size=args.batch_size, shuffle=True)
    net = UNet(args.num_classes).to(device)

    if os.path.exists(args.weight_path):
        net.load_state_dict(torch.load(args.weight_path))
        print_color("✅ Successfully loaded model weights!", "green")
    else:
        print_color("❌ No existing weights found. Training from scratch.", "red")

    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        print_color(f"Epoch [{epoch}/{args.epochs}]", "blue")
        show_gpu_info()

        loop = tqdm.tqdm(data_loader, desc="Training", ncols=100)
        for i, (image, segment_image) in enumerate(loop):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image.long())

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            loop.set_postfix(loss=f"{train_loss.item():.5f}")

            if i % args.save_interval == 0:
                _image = image[0]
                _segment_image = segment_image[0].unsqueeze(0).float() / 3.0
                _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0).float() / 3.0
                img = torch.cat([_segment_image, _out_image, _image[:1]], dim=0)

                os.makedirs(args.save_path, exist_ok=True)
                save_image(img, f'{args.save_path}/epoch{epoch}_batch{i}.png')

        if epoch % args.checkpoint_interval == 0:
            torch.save(net.state_dict(), args.weight_path)
            print_color(f"💾 Model saved at epoch {epoch}!", "green")

        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UNet Training Script")
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes (include background)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Total number of training epochs')
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
    parser.add_argument('--weight_path', type=str, default='params/unet.pth', help='Path to save/load model weights')
    parser.add_argument('--save_path', type=str, default='vessel_image', help='Path to save prediction images')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval (in batches) to save images')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Interval (in epochs) to save model checkpoint')

    args = parser.parse_args()
    main(args)
