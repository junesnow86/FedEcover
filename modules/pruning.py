import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet

from modules.models import CNN, DropoutScaling


def prune_linear_layer(linear_layer, pruned_indices: Dict[str, np.ndarray] = None):
    """
    Prune a linear layer by using provided pruned_indices to directly select neurons to drop.

    Parameters:
    - linear_layer: The linear layer to prune (an instance of torch.nn.Linear).
    - pruned_indices: A dictionary with keys 'input' and 'output', indicating the indices of neurons to prune directly.

    Returns:
    - new_layer: The new linear layer with pruned neurons.
    """
    assert isinstance(
        linear_layer, torch.nn.Linear
    ), "Input linear_layer must be an instance of torch.nn.Linear"

    input_features = linear_layer.in_features
    output_features = linear_layer.out_features

    if pruned_indices is not None:
        # Make sure the pruned indices are in relative order
        input_indices_to_keep = np.setdiff1d(
            range(input_features), pruned_indices.get("input", np.array([]))
        )
        output_indices_to_keep = np.setdiff1d(
            range(output_features), pruned_indices.get("output", np.array([]))
        )

    # Extract the weights and biases for the remaining neurons
    new_weight = linear_layer.weight.data[output_indices_to_keep, :][
        :, input_indices_to_keep
    ]
    new_bias = (
        linear_layer.bias.data[output_indices_to_keep]
        if linear_layer.bias is not None
        else None
    )

    # Create a new Linear layer with the pruned neurons
    new_layer = torch.nn.Linear(len(input_indices_to_keep), len(output_indices_to_keep))
    new_layer.weight.data = new_weight
    if new_bias is not None:
        new_layer.bias.data = new_bias

    return new_layer


def prune_conv_layer(conv_layer, pruned_indices: Dict[str, np.ndarray] = None):
    """
    Prune a convolution layer by using provided pruned_indices to directly select channels to drop.

    Parameters:
    - layer: The convolution layer to prune (an instance of torch.nn.Conv2d).
    - pruned_indices: A dictionary with keys 'input' and 'output', indicating the indices of channels to prune directly.

    Returns:
    - new_layer: The new convolution layer with pruned channels.
    """
    assert isinstance(
        conv_layer, torch.nn.Conv2d
    ), "Input layer must be an instance of torch.nn.Conv2d"

    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels

    if pruned_indices is not None:
        # Make sure the pruned indices are in relative order
        in_indices_to_keep = np.setdiff1d(
            range(in_channels), pruned_indices.get("input", np.array([]))
        )
        out_indices_to_keep = np.setdiff1d(
            range(out_channels), pruned_indices.get("output", np.array([]))
        )

    # Extract the weights and biases for the remaining filters
    new_weight = conv_layer.weight.data[out_indices_to_keep, :][
        :, in_indices_to_keep, :, :
    ]
    new_bias = (
        conv_layer.bias.data[out_indices_to_keep]
        if conv_layer.bias is not None
        else None
    )

    # Create a new Conv layer with the pruned filters
    new_conv_layer = torch.nn.Conv2d(
        len(in_indices_to_keep),
        len(out_indices_to_keep),
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=(conv_layer.bias is not None),
    )
    new_conv_layer.weight.data = new_weight
    if new_bias is not None:
        new_conv_layer.bias.data = new_bias

    return new_conv_layer


def prune_cnn(original_cnn: CNN, dropout_rate=0.5, scaling=True, **indices_to_prune):
    indices_to_prune_conv1 = indices_to_prune.get("indices_to_prune_conv1", None)
    indices_to_prune_conv2 = indices_to_prune.get("indices_to_prune_conv2", None)
    indices_to_prune_conv3 = indices_to_prune.get("indices_to_prune_conv3", None)
    indices_to_prune_fc = indices_to_prune.get("indices_to_prune_fc", None)

    conv1 = original_cnn.layer1[0]
    if indices_to_prune_conv1 is None:
        num_output_channels_to_prune_conv1 = int(conv1.out_channels * dropout_rate)
        output_indices_to_prune_conv1 = np.random.choice(
            conv1.out_channels, num_output_channels_to_prune_conv1, replace=False
        )
        indices_to_prune_conv1 = {"output": output_indices_to_prune_conv1}
    pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)

    conv2 = original_cnn.layer2[0]
    if indices_to_prune_conv2 is None:
        num_output_channels_to_prune_conv2 = int(conv2.out_channels * dropout_rate)
        output_indices_to_prune_conv2 = np.random.choice(
            conv2.out_channels, num_output_channels_to_prune_conv2, replace=False
        )
        indices_to_prune_conv2 = {
            "input": output_indices_to_prune_conv1,
            "output": output_indices_to_prune_conv2,
        }
    pruned_layer2 = prune_conv_layer(conv2, indices_to_prune_conv2)

    conv3 = original_cnn.layer3[0]
    if indices_to_prune_conv3 is None:
        num_output_channels_to_prune_conv3 = int(conv3.out_channels * dropout_rate)
        output_indices_to_prune_conv3 = np.random.choice(
            conv3.out_channels, num_output_channels_to_prune_conv3, replace=False
        )
        indices_to_prune_conv3 = {
            "input": output_indices_to_prune_conv2,
            "output": output_indices_to_prune_conv3,
        }
    pruned_layer3 = prune_conv_layer(conv3, indices_to_prune_conv3)

    fc = original_cnn.fc
    if indices_to_prune_fc is None:
        input_indices_to_prune_fc = []
        for channel_index in output_indices_to_prune_conv3:
            start_index = channel_index * 4 * 4
            end_index = (channel_index + 1) * 4 * 4
            input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
        input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
        indices_to_prune_fc = {"input": input_indices_to_prune_fc}
    pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)

    pruned_cnn = CNN()
    pruned_cnn.layer1[0] = pruned_layer1
    pruned_cnn.layer2[0] = pruned_layer2
    pruned_cnn.layer3[0] = pruned_layer3
    pruned_cnn.fc = pruned_fc

    if scaling:
        pruned_cnn.layer1.add_module("scaling", DropoutScaling(dropout_rate))
        pruned_cnn.layer2.add_module("scaling", DropoutScaling(dropout_rate))
        pruned_cnn.layer3.add_module("scaling", DropoutScaling(dropout_rate))

    return (
        pruned_cnn,
        indices_to_prune_conv1,
        indices_to_prune_conv2,
        indices_to_prune_conv3,
        indices_to_prune_fc,
    )


def create_random_even_groups(num_total_elements, num_groups):
    num_elements_per_group = num_total_elements // num_groups
    indices = np.arange(num_total_elements)
    np.random.shuffle(indices)
    indices = indices[: num_elements_per_group * num_groups]
    return np.array_split(indices, num_groups)


def select_random_group(groups):
    """
    Select a random group from the list of groups and remove it from the list.

    Parameters:
    - groups: A list of groups, where each group is a numpy array of indices.

    Returns:
    - selected_group: The selected group.
    - groups: The list of groups with the selected group removed.
    """
    group_index = np.random.choice(len(groups))
    selected_group = groups.pop(group_index)
    return selected_group, groups


def prune_cnn_into_groups(
    original_cnn: CNN,
    dropout_rate=0.5,
    scaling=True,
) -> Tuple[List[CNN], List[Dict]]:
    """
    Prune a CNN into multiple groups based on the dropout rate.

    Parameters:
    - original_cnn: The original CNN to prune.
    - dropout_rate: The dropout rate to use for pruning.
    - scaling: Whether to add a scaling layer after each pruned layer.

    Returns:
    - pruned_models: A list of pruned CNNs.
    - indices_to_prune_list: A list of dictionaries containing the indices to prune for each pruned CNN.
    """
    num_groups = max(int(1 / (1 - dropout_rate)), 1)
    pruned_models = []
    indices_to_prune_list = []

    conv1 = original_cnn.layer1[0]
    conv2 = original_cnn.layer2[0]
    conv3 = original_cnn.layer3[0]
    fc = original_cnn.fc

    groups_conv1 = create_random_even_groups(conv1.out_channels, num_groups)
    groups_conv2 = create_random_even_groups(conv2.out_channels, num_groups)
    groups_conv3 = create_random_even_groups(conv3.out_channels, num_groups)

    for _ in range(num_groups):
        group_conv1, groups_conv1 = select_random_group(groups_conv1)
        group_conv2, groups_conv2 = select_random_group(groups_conv2)
        group_conv3, groups_conv3 = select_random_group(groups_conv3)

        indices_to_prune_conv1 = {"output": group_conv1}
        indices_to_prune_conv2 = {
            "input": group_conv1,
            "output": group_conv2,
        }
        indices_to_prune_conv3 = {
            "input": group_conv2,
            "output": group_conv3,
        }

        input_indices_to_prune_fc = []
        for channel_index in group_conv3:
            start_index = channel_index * 4 * 4
            end_index = (channel_index + 1) * 4 * 4
            input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
        input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
        indices_to_prune_fc = {"input": input_indices_to_prune_fc}

        pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)
        pruned_layer2 = prune_conv_layer(conv2, indices_to_prune_conv2)
        pruned_layer3 = prune_conv_layer(conv3, indices_to_prune_conv3)
        pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)

        pruned_cnn = CNN()
        pruned_cnn.layer1[0] = pruned_layer1
        pruned_cnn.layer2[0] = pruned_layer2
        pruned_cnn.layer3[0] = pruned_layer3
        pruned_cnn.fc = pruned_fc

        if scaling:
            pruned_cnn.layer1.add_module("scaling", DropoutScaling(dropout_rate))
            pruned_cnn.layer2.add_module("scaling", DropoutScaling(dropout_rate))
            pruned_cnn.layer3.add_module("scaling", DropoutScaling(dropout_rate))

        pruned_models.append(pruned_cnn)
        indices_to_prune_list.append(
            {
                "indices_to_prune_conv1": indices_to_prune_conv1,
                "indices_to_prune_conv2": indices_to_prune_conv2,
                "indices_to_prune_conv3": indices_to_prune_conv3,
                "indices_to_prune_fc": indices_to_prune_fc,
            }
        )

    return pruned_models, indices_to_prune_list


def has_batchnorm_layer(model):
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            return True
    return False


def prune_resnet18(model, dropout_rate=0.5):
    """
    Prune a ResNet18 model by using the provided dropout rate.

    Parameters:
    - model: The ResNet18 model to prune.
    - dropout_rate: The dropout rate to use for pruning.

    Returns:
    - pruned_model: The pruned ResNet18 model.
    - pruned_indices_dict: A dictionary containing the indices to prune for each pruned layer.
    """
    # Note: using static layer normlization

    if not isinstance(model, ResNet):
        raise ValueError("Only ResNet18 is supported for now.")

    new_model = copy.deepcopy(model)

    # if has_batchnorm_layer(new_model):
    #     replace_bn_with_ln(new_model)
    #     print("BatchNorm layers replaced with LayerNorm layers.")

    pruned_indices_dict = {}

    layer_key = "conv1"
    layer = new_model.conv1
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {"output": out_channel_indices_to_prune}
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model, layer_key, new_layer)

    # Update layer norm's input shape
    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 16, 16], elementwise_affine=False
    )
    setattr(new_model, "bn1", new_layer_norm)

    # ----- layer1 -----
    former_layer_key = layer_key
    layer_key = "layer1.0.conv1"
    layer = new_model.layer1[0].conv1
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer1[0], "conv1", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 8, 8], elementwise_affine=False
    )
    setattr(new_model.layer1[0], "bn1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer1.0.conv2"
    layer = new_model.layer1[0].conv2
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer1[0], "conv2", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 8, 8], elementwise_affine=False
    )
    setattr(new_model.layer1[0], "bn2", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer1.1.conv1"
    layer = new_model.layer1[1].conv1
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer1[1], "conv1", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 8, 8], elementwise_affine=False
    )
    setattr(new_model.layer1[1], "bn1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer1.1.conv2"
    layer = new_model.layer1[1].conv2
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer1[1], "conv2", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 8, 8], elementwise_affine=False
    )
    setattr(new_model.layer1[1], "bn2", new_layer_norm)

    # ----- layer2 -----
    former_layer_key = layer_key
    layer_key = "layer2.0.conv1"
    layer = new_model.layer2[0].conv1
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer2[0], "conv1", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 4, 4], elementwise_affine=False
    )
    setattr(new_model.layer2[0], "bn1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer2.0.conv2"
    layer = new_model.layer2[0].conv2
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer2[0], "conv2", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 4, 4], elementwise_affine=False
    )
    setattr(new_model.layer2[0], "bn2", new_layer_norm)

    # a downsample layer's
    # input_indices_to_prune should be the same as the input_indices_to_prune of the first conv layer of the same block
    # output_indices_to_prune should be the same as the output_indices_to_prune of the last conv layer of the same block
    layer_key = "layer2.0.downsample.0"
    layer = new_model.layer2[0].downsample[0]
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict["layer2.0.conv1"]["input"],
        "output": pruned_indices_dict["layer2.0.conv2"]["output"],
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer2[0].downsample, "0", new_layer)

    num_out_channels_left = new_layer[0].out_channels
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 4, 4], elementwise_affine=False
    )
    setattr(new_model.layer2[0].downsample, "1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer2.1.conv1"
    layer = new_model.layer2[1].conv1
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer2[1], "conv1", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 4, 4], elementwise_affine=False
    )
    setattr(new_model.layer2[1], "bn1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer2.1.conv2"
    layer = new_model.layer2[1].conv2
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer2[1], "conv2", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 4, 4], elementwise_affine=False
    )
    setattr(new_model.layer2[1], "bn2", new_layer_norm)

    # ----- layer3 -----
    former_layer_key = layer_key
    layer_key = "layer3.0.conv1"
    layer = new_model.layer3[0].conv1
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer3[0], "conv1", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 2, 2], elementwise_affine=False
    )
    setattr(new_model.layer3[0], "bn1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer3.0.conv2"
    layer = new_model.layer3[0].conv2
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer3[0], "conv2", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 2, 2], elementwise_affine=False
    )
    setattr(new_model.layer3[0], "bn2", new_layer_norm)

    # a downsample layer's
    # input_indices_to_prune should be the same as the input_indices_to_prune of the first conv layer of the same block
    # output_indices_to_prune should be the same as the output_indices_to_prune of the last conv layer of the same block
    layer_key = "layer3.0.downsample.0"
    layer = new_model.layer3[0].downsample[0]
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict["layer3.0.conv1"]["input"],
        "output": pruned_indices_dict["layer3.0.conv2"]["output"],
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer3[0].downsample, "0", new_layer)

    num_out_channels_left = new_layer[0].out_channels
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 2, 2], elementwise_affine=False
    )
    setattr(new_model.layer3[0].downsample, "1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer3.1.conv1"
    layer = new_model.layer3[1].conv1
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer3[1], "conv1", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 2, 2], elementwise_affine=False
    )
    setattr(new_model.layer3[1], "bn1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer3.1.conv2"
    layer = new_model.layer3[1].conv2
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer3[1], "conv2", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 2, 2], elementwise_affine=False
    )
    setattr(new_model.layer3[1], "bn2", new_layer_norm)

    # ----- layer4 -----
    former_layer_key = layer_key
    layer_key = "layer4.0.conv1"
    layer = new_model.layer4[0].conv1
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer4[0], "conv1", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 1, 1], elementwise_affine=False
    )
    setattr(new_model.layer4[0], "bn1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer4.0.conv2"
    layer = new_model.layer4[0].conv2
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer4[0], "conv2", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 1, 1], elementwise_affine=False
    )
    setattr(new_model.layer4[0], "bn2", new_layer_norm)

    # a downsample layer's
    # input_indices_to_prune should be the same as the input_indices_to_prune of the first conv layer of the same block
    # output_indices_to_prune should be the same as the output_indices_to_prune of the last conv layer of the same block
    layer_key = "layer4.0.downsample.0"
    layer = new_model.layer4[0].downsample[0]
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict["layer4.0.conv1"]["input"],
        "output": pruned_indices_dict["layer4.0.conv2"]["output"],
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer4[0].downsample, "0", new_layer)

    num_out_channels_left = new_layer[0].out_channels
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 1, 1], elementwise_affine=False
    )
    setattr(new_model.layer4[0].downsample, "1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer4.1.conv1"
    layer = new_model.layer4[1].conv1
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer4[1], "conv1", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 1, 1], elementwise_affine=False
    )
    setattr(new_model.layer4[1], "bn1", new_layer_norm)

    former_layer_key = layer_key
    layer_key = "layer4.1.conv2"
    layer = new_model.layer4[1].conv2
    num_out_channels = layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dict[layer_key] = {
        "input": pruned_indices_dict[former_layer_key]["output"],
        "output": out_channel_indices_to_prune,
    }
    new_layer = prune_conv_layer(layer, pruned_indices_dict[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model.layer4[1], "conv2", new_layer)

    num_out_channels_left = num_out_channels - num_out_channels_to_prune
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 1, 1], elementwise_affine=False
    )
    setattr(new_model.layer4[1], "bn2", new_layer_norm)

    # ----- fc -----
    layer_key = "fc"
    layer = new_model.fc
    in_features_to_prune = np.sort(
        out_channel_indices_to_prune
    )  # the last conv layer's output indices
    # Since the last conv output H, W is 1, 1, we can just use the channel indices
    pruned_indices_dict[layer_key] = {"input": in_features_to_prune}
    new_layer = prune_linear_layer(layer, pruned_indices_dict[layer_key])
    setattr(new_model, layer_key, new_layer)

    return new_model, pruned_indices_dict
