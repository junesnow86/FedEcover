import torch
import torch.nn as nn
import torchvision.models as models

# 创建一个与CIFAR-10数据集形状相同的tensor
input_tensor = torch.randn(
    128, 3, 32, 32
)  # CIFAR-10: batch_size=1, channels=3, height=32, width=32

# 加载预训练的ResNet18模型
resnet18 = models.resnet18(weights=None)

# 用于存储每一层卷积层的输出形状
conv_output_shapes = {}


# 定义钩子函数
def hook_fn(module, input, output):
    for name, layer in resnet18.named_modules():
        if layer is module:
            conv_output_shapes[name] = output.shape
            break


# 注册钩子到每一个卷积层
hooks = []
for name, layer in resnet18.named_modules():
    if isinstance(layer, nn.Conv2d):
        hooks.append(layer.register_forward_hook(hook_fn))

# 前向传递
resnet18(input_tensor)

# 打印每一层卷积层的输出形状
for name, shape in conv_output_shapes.items():
    print(f"{name}: {shape}")

# 移除钩子
for hook in hooks:
    hook.remove()
