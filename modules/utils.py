import time

import numpy as np
import torch.nn as nn
from torchvision.models import ResNet


def calculate_model_size(model, print_result=True, unit="MB"):
    total_params = 0
    for param in model.parameters():
        # Multiply the size of each dimension to get the total number of elements
        total_params += param.numel()

    # Assuming each parameter is a 32-bit float
    memory_bytes = total_params * 4
    memory_kilobytes = memory_bytes / 1024
    memory_megabytes = memory_kilobytes / 1024

    if print_result:
        print(f"Total parameters: {total_params}")
        print(
            f"Memory Usage: {memory_bytes} bytes ({memory_kilobytes:.2f} KB / {memory_megabytes:.2f} MB)"
        )

    if unit == "KB":
        return memory_kilobytes
    elif unit == "MB":
        return memory_megabytes
    else:
        return memory_bytes


def replace_bn_with_ln(model: nn.Module, affine=False, dataset="cifar10"):
    """
    Replace all BatchNorm layers in the model with LayerNorm layers.

    ResNet18 model has the following structure:
    conv1: torch.Size([128, 64, 16, 16])
    layer1.0.conv1: torch.Size([128, 64, 8, 8])
    layer1.0.conv2: torch.Size([128, 64, 8, 8])
    layer1.1.conv1: torch.Size([128, 64, 8, 8])
    layer1.1.conv2: torch.Size([128, 64, 8, 8])
    layer2.0.conv1: torch.Size([128, 128, 4, 4])
    layer2.0.conv2: torch.Size([128, 128, 4, 4])
    layer2.0.downsample.0: torch.Size([128, 128, 4, 4])
    layer2.1.conv1: torch.Size([128, 128, 4, 4])
    layer2.1.conv2: torch.Size([128, 128, 4, 4])
    layer3.0.conv1: torch.Size([128, 256, 2, 2])
    layer3.0.conv2: torch.Size([128, 256, 2, 2])
    layer3.0.downsample.0: torch.Size([128, 256, 2, 2])
    layer3.1.conv1: torch.Size([128, 256, 2, 2])
    layer3.1.conv2: torch.Size([128, 256, 2, 2])
    layer4.0.conv1: torch.Size([128, 512, 1, 1])
    layer4.0.conv2: torch.Size([128, 512, 1, 1])
    layer4.0.downsample.0: torch.Size([128, 512, 1, 1])
    layer4.1.conv1: torch.Size([128, 512, 1, 1])
    layer4.1.conv2: torch.Size([128, 512, 1, 1])
    """
    if not isinstance(model, ResNet):
        raise ValueError("Only ResNet18 is supported for now.")

    if dataset == "cifar10" or dataset == "cifar100":
        layernorm_shapes = {
            "bn1": [64, 16, 16],
            "layer1": [64, 8, 8],
            "layer2": [128, 4, 4],
            "layer3": [256, 2, 2],
            "layer4": [512, 1, 1],
        }
    elif dataset == "tiny-imagenet":
        print("Using Tiny ImageNet layernorm shapes")
        layernorm_shapes = {
            "bn1": [64, 32, 32],
            "layer1": [64, 16, 16],
            "layer2": [128, 8, 8],
            "layer3": [256, 4, 4],
            "layer4": [512, 2, 2],
        }

    # Replace all BatchNorm layers with LayerNorm layers
    model.bn1 = nn.LayerNorm(layernorm_shapes["bn1"], elementwise_affine=affine)

    model.layer1[0].bn1 = nn.LayerNorm(layernorm_shapes["layer1"], elementwise_affine=affine)
    model.layer1[0].bn2 = nn.LayerNorm(layernorm_shapes["layer1"], elementwise_affine=affine)
    model.layer1[1].bn1 = nn.LayerNorm(layernorm_shapes["layer1"], elementwise_affine=affine)
    model.layer1[1].bn2 = nn.LayerNorm(layernorm_shapes["layer1"], elementwise_affine=affine)

    model.layer2[0].bn1 = nn.LayerNorm(layernorm_shapes["layer2"], elementwise_affine=affine)
    model.layer2[0].bn2 = nn.LayerNorm(layernorm_shapes["layer2"], elementwise_affine=affine)
    model.layer2[0].downsample[1] = nn.LayerNorm(layernorm_shapes["layer2"], elementwise_affine=affine)
    model.layer2[1].bn1 = nn.LayerNorm(layernorm_shapes["layer2"], elementwise_affine=affine)
    model.layer2[1].bn2 = nn.LayerNorm(layernorm_shapes["layer2"], elementwise_affine=affine)

    model.layer3[0].bn1 = nn.LayerNorm(layernorm_shapes["layer3"], elementwise_affine=affine)
    model.layer3[0].bn2 = nn.LayerNorm(layernorm_shapes["layer3"], elementwise_affine=affine)
    model.layer3[0].downsample[1] = nn.LayerNorm(layernorm_shapes["layer3"], elementwise_affine=affine)
    model.layer3[1].bn1 = nn.LayerNorm(layernorm_shapes["layer3"], elementwise_affine=affine)
    model.layer3[1].bn2 = nn.LayerNorm(layernorm_shapes["layer3"], elementwise_affine=affine)

    model.layer4[0].bn1 = nn.LayerNorm(layernorm_shapes["layer4"], elementwise_affine=affine)
    model.layer4[0].bn2 = nn.LayerNorm(layernorm_shapes["layer4"], elementwise_affine=affine)
    model.layer4[0].downsample[1] = nn.LayerNorm(layernorm_shapes["layer4"], elementwise_affine=affine)
    model.layer4[1].bn1 = nn.LayerNorm(layernorm_shapes["layer4"], elementwise_affine=affine)
    model.layer4[1].bn2 = nn.LayerNorm(layernorm_shapes["layer4"], elementwise_affine=affine)


def replace_bn_with_sbn(model: nn.Module):
    """
    Replace all BatchNorm layers in the model with static BatchNorm layers.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m = nn.BatchNorm2d(m.num_features, affine=False)


def measure_time(repeats: int = 1):
    """Decorator, measure the function time costs with mean and variance.

    Args:
        repeats (int, optional): Repeat times for measuring. Defaults to 10.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            timings = []
            for _ in range(repeats):
                start_time = time.time()
                func_res = func(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                timings.append(elapsed)
            np_times = np.array(timings)
            average_time = np.mean(np_times)
            variance = np.var(np_times)
            # logging.info(f"[{func.__name__}] Average time over {repeats} runs: {average_time:.6f} seconds")
            print(
                f"[{func.__name__}] Average time over {repeats} runs: {average_time:.6f} seconds"
            )
            if repeats > 1:
                # logging.info(f"[{func.__name__}] Variance of times: {variance:.6f}")
                print(f"[{func.__name__}] Variance of times: {variance:.6f}")
            return func_res

        return wrapper

    return decorator
