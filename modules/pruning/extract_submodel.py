import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet

from modules.models import CNN, DropoutScaling

from .submodel_param_indices_dicts import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
)


def extract_sublayer_linear(
    original_layer: nn.Linear,
    layer_param_indices_dict: SubmodelLayerParamIndicesDict,
):
    in_features = original_layer.in_features
    out_features = original_layer.out_features

    in_indices, out_indices = (
        layer_param_indices_dict.get("in", np.array(range(in_features))),
        layer_param_indices_dict.get("out", np.array(range(out_features))),
    )
    # Check in_indices and out_indices are sorted
    assert np.all(in_indices == np.sort(in_indices))
    assert np.all(out_indices == np.sort(out_indices))

    sublayer_weight = torch.index_select(
        torch.index_select(original_layer.weight.data, 0, torch.tensor(out_indices)),
        1,
        torch.tensor(in_indices),
    )
    sublayer_bias = (
        torch.index_select(original_layer.bias.data, 0, torch.tensor(out_indices))
        if original_layer.bias is not None
        else None
    )

    new_layer = nn.Linear(
        len(in_indices), len(out_indices), bias=sublayer_bias is not None
    )
    with torch.no_grad():
        new_layer.weight.data.copy_(sublayer_weight)
        if sublayer_bias is not None:
            new_layer.bias.data.copy_(sublayer_bias)

    return new_layer


def extract_sublayer_conv2d(
    original_layer: nn.Conv2d,
    layer_param_indices_dict: SubmodelLayerParamIndicesDict,
):
    in_channels = original_layer.in_channels
    out_channels = original_layer.out_channels
    kernel_size = original_layer.kernel_size
    stride = original_layer.stride
    padding = original_layer.padding
    dilation = original_layer.dilation
    groups = original_layer.groups

    in_indices, out_indices = (
        layer_param_indices_dict.get("in", np.array(range(in_channels))),
        layer_param_indices_dict.get("out", np.array(range(out_channels))),
    )
    # Check in_indices and out_indices are sorted
    assert np.all(in_indices == np.sort(in_indices))
    assert np.all(out_indices == np.sort(out_indices))

    sublayer_weight = torch.index_select(
        torch.index_select(original_layer.weight.data, 0, torch.tensor(out_indices)),
        1,
        torch.tensor(in_indices),
    )
    sublayer_bias = (
        torch.index_select(original_layer.bias.data, 0, torch.tensor(out_indices))
        if original_layer.bias is not None
        else None
    )

    new_layer = nn.Conv2d(
        len(in_indices),
        len(out_indices),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=sublayer_bias is not None,
    )
    with torch.no_grad():
        new_layer.weight.data.copy_(sublayer_weight)
        if sublayer_bias is not None:
            new_layer.bias.data.copy_(sublayer_bias)

    return new_layer


def extract_submodel_cnn(
    original_model: CNN,
    submodel_param_indices_dict: SubmodelBlockParamIndicesDict,
    p: float,
    scaling: bool = True,
):
    submodel = CNN()

    sublayer_conv1 = extract_sublayer_conv2d(
        original_model.layer1[0],
        submodel_param_indices_dict["layer1.0"],
    )
    if scaling:
        sublayer_conv1 = nn.Sequential(sublayer_conv1, DropoutScaling(p=p))
    else:
        sublayer_conv1 = nn.Sequential(sublayer_conv1)
    submodel.layer1[0] = sublayer_conv1

    sublayer_conv2 = extract_sublayer_conv2d(
        original_model.layer2[0],
        submodel_param_indices_dict["layer2.0"],
    )
    if scaling:
        sublayer_conv2 = nn.Sequential(sublayer_conv2, DropoutScaling(p=p))
    else:
        sublayer_conv2 = nn.Sequential(sublayer_conv2)
    submodel.layer2[0] = sublayer_conv2

    sublayer_conv3 = extract_sublayer_conv2d(
        original_model.layer3[0],
        submodel_param_indices_dict["layer3.0"],
    )
    if scaling:
        sublayer_conv3 = nn.Sequential(sublayer_conv3, DropoutScaling(p=p))
    else:
        sublayer_conv3 = nn.Sequential(sublayer_conv3)
    submodel.layer3[0] = sublayer_conv3

    sublayer_fc = extract_sublayer_linear(
        original_model.fc,
        submodel_param_indices_dict["fc"],
    )
    submodel.fc = sublayer_fc

    return submodel


def extract_submodel_resnet(
    original_model: ResNet,
    submodel_param_indices_dict: SubmodelBlockParamIndicesDict,
    p: float,
    scaling: bool = True,
    dataset: str = "cifar10",
    norm_type: str = "ln",
):
    submodel = copy.deepcopy(original_model)

    sublayer_conv1 = extract_sublayer_conv2d(
        original_model.conv1,
        submodel_param_indices_dict["conv1"],
    )
    if scaling:
        sublayer_conv1 = nn.Sequential(sublayer_conv1, DropoutScaling(p=p))
    else:
        sublayer_conv1 = nn.Sequential(sublayer_conv1)
    submodel.conv1 = sublayer_conv1
    if norm_type == "ln":
        if dataset == "cifar10" or dataset == "cifar100":
            submodel.bn1 = nn.LayerNorm(
                normalized_shape=[submodel.conv1[0].out_channels, 16, 16],
                elementwise_affine=False,
            )
        elif dataset == "tiny-imagenet":
            submodel.bn1 = nn.LayerNorm(
                normalized_shape=[submodel.conv1[0].out_channels, 32, 32],
                elementwise_affine=False,
            )
        else:
            raise ValueError("Invalid dataset")
    elif norm_type == "sbn":
        submodel.bn1 = nn.BatchNorm2d(submodel.conv1[0].out_channels, affine=False, track_running_stats=False)
    else:
        raise ValueError("Invalid norm_type")

    layers = ["layer1", "layer2", "layer3", "layer4"]
    if norm_type == "ln":
        if dataset == "cifar10" or dataset == "cifar100":
            layernorm_shapes = [
                [64, 8, 8],
                [128, 4, 4],
                [256, 2, 2],
                [512, 1, 1],
            ]
        elif dataset == "tiny-imagenet":
            layernorm_shapes = [
                [64, 16, 16],
                [128, 8, 8],
                [256, 4, 4],
                [512, 2, 2],
            ]
        else:
            raise ValueError("Invalid dataset")
    blocks = ["0", "1"]
    convs = ["conv1", "conv2"]

    for i, layer in enumerate(layers):
        for block in blocks:
            for conv in convs:
                sublayer_conv = extract_sublayer_conv2d(
                    original_model._modules[layer][int(block)]._modules[conv],
                    submodel_param_indices_dict[f"{layer}.{block}.{conv}"],
                )
                if scaling:
                    sublayer_conv = nn.Sequential(sublayer_conv, DropoutScaling(p=p))
                else:
                    sublayer_conv = nn.Sequential(sublayer_conv)
                submodel._modules[layer][int(block)]._modules[conv] = sublayer_conv
                if norm_type == "ln":
                    layernorm_shape = layernorm_shapes[i]
                    layernorm_shape[0] = sublayer_conv[0].out_channels
                    submodel._modules[layer][int(block)]._modules[f"bn{int(conv[-1])}"] = (
                        nn.LayerNorm(
                            normalized_shape=layernorm_shape, elementwise_affine=False
                        )
                    )
                elif norm_type == "sbn":
                    submodel._modules[layer][int(block)]._modules[f"bn{int(conv[-1])}"] = (
                        nn.BatchNorm2d(sublayer_conv[0].out_channels, affine=False, track_running_stats=False)
                    )
                else:
                    raise ValueError("Invalid norm_type")

            if layer != "layer1" and block == "0":
                sublayer_downsample = extract_sublayer_conv2d(
                    original_model._modules[layer][int(block)].downsample[0],
                    submodel_param_indices_dict[f"{layer}.{block}.downsample.0"],
                )
                if scaling:
                    sublayer_downsample = nn.Sequential(
                        sublayer_downsample, DropoutScaling(p=p)
                    )
                else:
                    sublayer_downsample = nn.Sequential(sublayer_downsample)
                submodel._modules[layer][int(block)].downsample[0] = sublayer_downsample
                if norm_type == "ln":
                    layernorm_shape = layernorm_shapes[i]
                    layernorm_shape[0] = sublayer_downsample[0].out_channels
                    submodel._modules[layer][int(block)].downsample[1] = nn.LayerNorm(
                        normalized_shape=layernorm_shape, elementwise_affine=False
                    )
                elif norm_type == "sbn":
                    submodel._modules[layer][int(block)].downsample[1] = (
                        nn.BatchNorm2d(sublayer_downsample[0].out_channels, affine=False, track_running_stats=False)
                    )
                else:
                    raise ValueError("Invalid norm_type")

    sublayer_fc = extract_sublayer_linear(
        original_model.fc,
        submodel_param_indices_dict["fc"],
    )
    submodel.fc = sublayer_fc

    return submodel


def extract_submodel_update_direction(
    global_update_direction: Dict[str, torch.Tensor],
    submodel_param_indices_dict: SubmodelBlockParamIndicesDict,
):
    submodel_update_direction = {}

    for name, update_direction in global_update_direction.items():
        layer_name = name.replace(".weight", "").replace(".bias", "")

        in_indices, out_indices = (
            submodel_param_indices_dict[layer_name].get("in", None),
            submodel_param_indices_dict[layer_name].get("out", None),
        )
        # Check in_indices and out_indices are sorted
        if in_indices is not None:
            assert np.all(in_indices == np.sort(in_indices))
        if out_indices is not None:
            assert np.all(out_indices == np.sort(out_indices))

        if "weight" in name:
            sub_update_direction = torch.index_select(
                torch.index_select(update_direction, 0, torch.tensor(out_indices)),
                1,
                torch.tensor(in_indices),
            )
        elif "bias" in name:
            sub_update_direction = torch.index_select(
                update_direction, 0, torch.tensor(out_indices)
            )
        else:
            raise ValueError("Invalid name")

        submodel_update_direction[name] = sub_update_direction

    return submodel_update_direction


def extract_param_control_variate(
    name: str,
    param_control_variate: torch.Tensor,
    layer_param_indices_dict: SubmodelLayerParamIndicesDict,
):
    if "weight" in name:
        in_features = param_control_variate.size(1)
        in_indices = layer_param_indices_dict.get("in", np.array(range(in_features)))

    out_features = param_control_variate.size(0)
    out_indices = layer_param_indices_dict.get("out", np.array(range(out_features)))

    if "weight" in name:
        sub_param_control_variate = torch.index_select(
            torch.index_select(param_control_variate, 0, torch.tensor(out_indices)),
            1,
            torch.tensor(in_indices),
        )
    elif "bias" in name:
        sub_param_control_variate = torch.index_select(
            param_control_variate, 0, torch.tensor(out_indices)
        )
    else:
        raise ValueError("Invalid name")

    return sub_param_control_variate


def extract_submodel_control_variates(
    complete_control_variates: Dict[str, torch.Tensor],
    submodel_param_indices_dict: SubmodelBlockParamIndicesDict,
):
    submodel_control_variates = {}

    for name, param_control_variate in complete_control_variates.items():
        key = name.replace(".weight", "").replace(".bias", "")
        submodel_control_variates[name] = extract_param_control_variate(
            name,
            param_control_variate,
            submodel_param_indices_dict[key],
        )

    return submodel_control_variates
