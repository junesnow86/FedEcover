import numpy as np
import torch
import torch.nn as nn

from .submodel_param_indices_dicts import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
)
from modules.models import CNN, DropoutScaling


def extract_sublayer_linear(
    original_layer: nn.Linear,
    layer_param_indices_dict: SubmodelLayerParamIndicesDict,
):
    in_features = original_layer.in_features
    out_features = original_layer.out_features

    # Check in_indices and out_indices are sorted
    in_indices, out_indices = (
        layer_param_indices_dict.get("in", np.array(range(in_features))),
        layer_param_indices_dict.get("out", np.array(range(out_features))),
    )
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

    # Check in_indices and out_indices are sorted
    in_indices, out_indices = (
        layer_param_indices_dict.get("in", np.array(range(in_channels))),
        layer_param_indices_dict.get("out", np.array(range(out_channels))),
    )
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
    submodel.layer1[0] = sublayer_conv1

    sublayer_conv2 = extract_sublayer_conv2d(
        original_model.layer2[0],
        submodel_param_indices_dict["layer2.0"],
    )
    if scaling:
        sublayer_conv2 = nn.Sequential(sublayer_conv2, DropoutScaling(p=p))
    submodel.layer2[0] = sublayer_conv2

    sublayer_conv3 = extract_sublayer_conv2d(
        original_model.layer3[0],
        submodel_param_indices_dict["layer3.0"],
    )
    if scaling:
        sublayer_conv3 = nn.Sequential(sublayer_conv3, DropoutScaling(p=p))
    submodel.layer3[0] = sublayer_conv3

    sublayer_fc = extract_sublayer_linear(
        original_model.fc,
        submodel_param_indices_dict["fc"],
    )
    submodel.fc = sublayer_fc

    return submodel
