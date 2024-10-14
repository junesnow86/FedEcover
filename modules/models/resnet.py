import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet


def replace_bn_with_ln(model: nn.Module, affine=False, input_shape=None):
    """
    Replace all BatchNorm layers in the resnet18 model with not-affine LayerNorm layers. In-place operation.

    ResNet18 model on CIFAR datasets has the following structure:
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

    if input_shape is None:
        input_shape = [32, 32]

    assert input_shape[0] == input_shape[1]
    assert input_shape[0] in [32, 64]

    if input_shape[0] == 32:
        layernorm_shapes = {
            "bn1": [64, 16, 16],
            "layer1": [64, 8, 8],
            "layer2": [128, 4, 4],
            "layer3": [256, 2, 2],
            "layer4": [512, 1, 1],
        }
    else:
        layernorm_shapes = {
            "bn1": [64, 32, 32],
            "layer1": [64, 16, 16],
            "layer2": [128, 8, 8],
            "layer3": [256, 4, 4],
            "layer4": [512, 2, 2],
        }

    model.bn1 = nn.LayerNorm(layernorm_shapes["bn1"], elementwise_affine=affine)

    model.layer1[0].bn1 = nn.LayerNorm(
        layernorm_shapes["layer1"], elementwise_affine=affine
    )
    model.layer1[0].bn2 = nn.LayerNorm(
        layernorm_shapes["layer1"], elementwise_affine=affine
    )
    model.layer1[1].bn1 = nn.LayerNorm(
        layernorm_shapes["layer1"], elementwise_affine=affine
    )
    model.layer1[1].bn2 = nn.LayerNorm(
        layernorm_shapes["layer1"], elementwise_affine=affine
    )

    model.layer2[0].bn1 = nn.LayerNorm(
        layernorm_shapes["layer2"], elementwise_affine=affine
    )
    model.layer2[0].bn2 = nn.LayerNorm(
        layernorm_shapes["layer2"], elementwise_affine=affine
    )
    model.layer2[0].downsample[1] = nn.LayerNorm(
        layernorm_shapes["layer2"], elementwise_affine=affine
    )
    model.layer2[1].bn1 = nn.LayerNorm(
        layernorm_shapes["layer2"], elementwise_affine=affine
    )
    model.layer2[1].bn2 = nn.LayerNorm(
        layernorm_shapes["layer2"], elementwise_affine=affine
    )

    model.layer3[0].bn1 = nn.LayerNorm(
        layernorm_shapes["layer3"], elementwise_affine=affine
    )
    model.layer3[0].bn2 = nn.LayerNorm(
        layernorm_shapes["layer3"], elementwise_affine=affine
    )
    model.layer3[0].downsample[1] = nn.LayerNorm(
        layernorm_shapes["layer3"], elementwise_affine=affine
    )
    model.layer3[1].bn1 = nn.LayerNorm(
        layernorm_shapes["layer3"], elementwise_affine=affine
    )
    model.layer3[1].bn2 = nn.LayerNorm(
        layernorm_shapes["layer3"], elementwise_affine=affine
    )

    model.layer4[0].bn1 = nn.LayerNorm(
        layernorm_shapes["layer4"], elementwise_affine=affine
    )
    model.layer4[0].bn2 = nn.LayerNorm(
        layernorm_shapes["layer4"], elementwise_affine=affine
    )
    model.layer4[0].downsample[1] = nn.LayerNorm(
        layernorm_shapes["layer4"], elementwise_affine=affine
    )
    model.layer4[1].bn1 = nn.LayerNorm(
        layernorm_shapes["layer4"], elementwise_affine=affine
    )
    model.layer4[1].bn2 = nn.LayerNorm(
        layernorm_shapes["layer4"], elementwise_affine=affine
    )


def replace_bn_with_sbn(model: nn.Module, affine=False):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            sbn = nn.BatchNorm2d(module.num_features, affine=affine, track_running_stats=False)
            parent_module = model
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], sbn)


def custom_resnet18(num_classes, weights=None, norm_type="ln", input_shape=None):
    model = resnet18(num_classes=num_classes, weights=weights)
    if norm_type == "ln":
        replace_bn_with_ln(model, affine=False, input_shape=input_shape)
    elif norm_type == "sbn":
        replace_bn_with_sbn(model, affine=False)
    else:
        raise ValueError(f"Normalization type {norm_type} not supported.")
    return model


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ShallowResNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            # BasicBlock(64, 64),
        )

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            # BasicBlock(128, 128),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        # self.fc = nn.Linear(64, num_classes)

        self.layer1[0].bn1 = nn.LayerNorm([64, 8, 8], elementwise_affine=False)
        self.layer1[0].bn2 = nn.LayerNorm([64, 8, 8], elementwise_affine=False)
        # self.layer1[1].bn1 = nn.LayerNorm([64, 8, 8], elementwise_affine=False)
        # self.layer1[1].bn2 = nn.LayerNorm([64, 8, 8], elementwise_affine=False)
        self.layer2[0].bn1 = nn.LayerNorm([128, 4, 4], elementwise_affine=False)
        self.layer2[0].bn2 = nn.LayerNorm([128, 4, 4], elementwise_affine=False)
        self.layer2[0].downsample[1] = nn.LayerNorm(
            [128, 4, 4], elementwise_affine=False
        )
        # self.layer2[1].bn1 = nn.LayerNorm([128, 4, 4], elementwise_affine=False)
        # self.layer2[1].bn2 = nn.LayerNorm([128, 4, 4], elementwise_affine=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
