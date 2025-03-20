import torch
from net import UNet

def export():
    model = UNet(num_classes=4)
    model.load_state_dict(torch.load("params/unet.pth")) 
    dummy_input = torch.randn(1, 3, 512, 512)
    
    torch.onnx.export(model, dummy_input, "model.onnx",
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=12,
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

if __name__ == "__main__":
    export()