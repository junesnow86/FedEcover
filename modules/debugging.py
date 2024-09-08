import torch.nn as nn


def create_empty_pruned_indices_dict():
    pruned_indices_dict = {
        "conv1": {},
        "layer1.0.conv1": {},
        "layer1.0.conv2": {},
        "layer1.1.conv1": {},
        "layer1.1.conv2": {},
        "layer2.0.conv1": {},
        "layer2.0.conv2": {},
        "layer2.0.downsample.0": {},
        "layer2.1.conv1": {},
        "layer2.1.conv2": {},
        "layer3.0.conv1": {},
        "layer3.0.conv2": {},
        "layer3.0.downsample.0": {},
        "layer3.1.conv1": {},
        "layer3.1.conv2": {},
        "layer4.0.conv1": {},
        "layer4.0.conv2": {},
        "layer4.0.downsample.0": {},
        "layer4.1.conv1": {},
        "layer4.1.conv2": {},
        "fc": {},
    }
    return pruned_indices_dict


def replace_bn_with_identity(model: nn.Module, affine=False):
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

    # Replace all BatchNorm layers with LayerNorm layers
    model.bn1 = nn.Identity()

    model.layer1[0].bn1 = nn.Identity()
    model.layer1[0].bn2 = nn.Identity()
    model.layer1[1].bn1 = nn.Identity()
    model.layer1[1].bn2 = nn.Identity()

    model.layer2[0].bn1 = nn.Identity()
    model.layer2[0].bn2 = nn.Identity()
    model.layer2[0].downsample[1] = nn.Identity()
    model.layer2[1].bn1 = nn.Identity()
    model.layer2[1].bn2 = nn.Identity()

    model.layer3[0].bn1 = nn.Identity()
    model.layer3[0].bn2 = nn.Identity()
    model.layer3[0].downsample[1] = nn.Identity()
    model.layer3[1].bn1 = nn.Identity()
    model.layer3[1].bn2 = nn.Identity()

    model.layer4[0].bn1 = nn.Identity()
    model.layer4[0].bn2 = nn.Identity()
    model.layer4[0].downsample[1] = nn.Identity()
    model.layer4[1].bn1 = nn.Identity()
    model.layer4[1].bn2 = nn.Identity()
