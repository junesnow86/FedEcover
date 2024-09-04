import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet

from modules.models import CNN, DropoutScaling


def create_indices_bags(num_elements: int, bag_size: int) -> List[np.ndarray]:
    assert num_elements >= bag_size, "Number of elements must be greater than bag size."

    all_indices = np.arange(num_elements)
    np.random.shuffle(all_indices)

    indices_bags = []
    total_collected = 0
    while total_collected < num_elements:
        if num_elements - total_collected < bag_size:
            num_remaining = num_elements - total_collected
            num_additional_needed = bag_size - num_remaining
            indices_bag = np.concatenate(
                [all_indices[total_collected:], all_indices[:num_additional_needed]]
            )
            total_collected += num_remaining
        else:
            indices_bag = all_indices[total_collected : total_collected + bag_size]
            total_collected += bag_size

        indices_bags.append(indices_bag)

    return indices_bags


def pruned_indices_dict_bagging_cnn(dropout_rate: float):
    """Pack the indices to prune for each layer of a CNN model into bags.

    Args:
        dropout_rate: The dropout rate to use for pruning.

    Returns:
        pruned_indices_bags: A list of dictionaries containing the indices to prune for each layer.
    """
    pruned_indices_bags = []

    # conv1
    num_output_channels_conv1 = 64
    num_output_channels_to_prune_conv1 = int(num_output_channels_conv1 * dropout_rate)
    num_output_channels_keep_conv1 = (
        num_output_channels_conv1 - num_output_channels_to_prune_conv1
    )
    keep_indices_bags_conv1 = create_indices_bags(
        num_output_channels_conv1, num_output_channels_keep_conv1
    )
    pruned_indices_bags_conv1 = [
        np.sort(np.setdiff1d(np.arange(num_output_channels_conv1), keep_indices_bag))
        for keep_indices_bag in keep_indices_bags_conv1
    ]

    # conv2
    num_output_channels_conv2 = 128
    num_output_channels_to_prune_conv2 = int(num_output_channels_conv2 * dropout_rate)
    num_output_channels_keep_conv2 = (
        num_output_channels_conv2 - num_output_channels_to_prune_conv2
    )
    keep_indices_bags_conv2 = create_indices_bags(
        num_output_channels_conv2, num_output_channels_keep_conv2
    )
    pruned_indices_bags_conv2 = [
        np.sort(np.setdiff1d(np.arange(num_output_channels_conv2), keep_indices_bag))
        for keep_indices_bag in keep_indices_bags_conv2
    ]

    # conv3
    num_output_channels_conv3 = 256
    num_output_channels_to_prune_conv3 = int(num_output_channels_conv3 * dropout_rate)
    num_output_channels_keep_conv3 = (
        num_output_channels_conv3 - num_output_channels_to_prune_conv3
    )
    keep_indices_bags_conv3 = create_indices_bags(
        num_output_channels_conv3, num_output_channels_keep_conv3
    )
    pruned_indices_bags_conv3 = [
        np.sort(np.setdiff1d(np.arange(num_output_channels_conv3), keep_indices_bag))
        for keep_indices_bag in keep_indices_bags_conv3
    ]

    # Make sure the length of the bags are the same
    min_length = min(
        len(pruned_indices_bags_conv1),
        len(pruned_indices_bags_conv2),
        len(pruned_indices_bags_conv3),
    )
    pruned_indices_bags_conv1 = pruned_indices_bags_conv1[:min_length]
    pruned_indices_bags_conv2 = pruned_indices_bags_conv2[:min_length]
    pruned_indices_bags_conv3 = pruned_indices_bags_conv3[:min_length]

    # fc
    H = 4
    W = 4
    pruned_indices_bags_fc = []
    for bag in pruned_indices_bags_conv3:
        input_indices_to_prune_fc = []
        for channel_index in bag:
            start_index = channel_index * H * W
            end_index = (channel_index + 1) * H * W
            input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
        input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
        pruned_indices_bags_fc.append(input_indices_to_prune_fc)

    num_bags = min_length
    for i in range(num_bags):
        pruned_indices_bag = {
            "layer1": {"output": pruned_indices_bags_conv1[i]},
            "layer2": {
                "input": pruned_indices_bags_conv1[i],
                "output": pruned_indices_bags_conv2[i],
            },
            "layer3": {
                "input": pruned_indices_bags_conv2[i],
                "output": pruned_indices_bags_conv3[i],
            },
            "fc": {"input": pruned_indices_bags_fc[i]},
        }
        pruned_indices_bags.append(pruned_indices_bag)

    return pruned_indices_bags


def pruned_indices_dict_bagging_resnet18(dropout_rate: float):
    """Pack the indices to prune for each layer of a resnet18 model into bags.

    Args:
        dropout_rate: The dropout rate to use for pruning.

    Returns:
        pruned_indices_bags: A list of dictionaries containing the indices to prune for each layer.
    """
    pruned_indices_dict_bags = []
    pruned_out_indices_bags_for_each_layer = {}

    num_out_channels = 64
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    num_out_channels_keep = num_out_channels - num_out_channels_to_prune
    keep_out_indices_bags = create_indices_bags(num_out_channels, num_out_channels_keep)
    pruned_out_indices_bags = [
        np.sort(np.setdiff1d(np.arange(num_out_channels), bag))
        for bag in keep_out_indices_bags
    ]
    pruned_out_indices_bags_for_each_layer["conv1"] = pruned_out_indices_bags

    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    out_channels_numbers = [64, 128, 256, 512]
    blocks = ["0", "1"]
    convs = ["conv1", "conv2"]

    for i, layer_name in enumerate(layer_names):
        num_out_channels = out_channels_numbers[i]
        num_out_channels_to_prune = int(num_out_channels * dropout_rate)
        num_out_channels_keep = num_out_channels - num_out_channels_to_prune
        for block in blocks:
            for conv in convs:
                layer_key = f"{layer_name}.{block}.{conv}"
                keep_out_indices_bags = create_indices_bags(
                    num_out_channels, num_out_channels_keep
                )
                pruned_out_indices_bags = [
                    np.sort(np.setdiff1d(np.arange(num_out_channels), bag))
                    for bag in keep_out_indices_bags
                ]
                pruned_out_indices_bags_for_each_layer[layer_key] = (
                    pruned_out_indices_bags
                )

    # Make sure the length of the bags are the same
    min_length = min(
        [
            len(pruned_out_indices_bags)
            for pruned_out_indices_bags in pruned_out_indices_bags_for_each_layer.values()
        ]
    )
    for key in pruned_out_indices_bags_for_each_layer.keys():
        pruned_out_indices_bags_for_each_layer[key] = (
            pruned_out_indices_bags_for_each_layer[key][:min_length]
        )

    # fc
    H = 1
    W = 1
    pruned_out_indices_bags = []
    for bag in pruned_out_indices_bags_for_each_layer["layer4.1.conv2"]:
        in_indices_to_prune = []
        for channel_index in bag:
            start_index = channel_index * H * W
            end_index = (channel_index + 1) * H * W
            in_indices_to_prune.extend(list(range(start_index, end_index)))
        in_indices_to_prune = np.sort(in_indices_to_prune)
        pruned_out_indices_bags.append(in_indices_to_prune)

    num_bags = min_length
    for i in range(num_bags):
        pruned_indices_dict_bag = {}
        # conv1
        pruned_indices_dict_bag["conv1"] = {
            "output": pruned_out_indices_bags_for_each_layer["conv1"][i]
        }

        # layers
        former_layer_key = "conv1"
        for j, layer_name in enumerate(layer_names):
            for block in blocks:
                for conv in convs:
                    layer_key = f"{layer_name}.{block}.{conv}"
                    pruned_indices_dict_bag[layer_key] = {
                        "input": pruned_out_indices_bags_for_each_layer[
                            former_layer_key
                        ][i],
                        "output": pruned_out_indices_bags_for_each_layer[layer_key][i],
                    }
                    former_layer_key = layer_key

                # If there is downsample layer
                if j > 0 and block == "0":
                    layer_key = f"{layer_name}.{block}.downsample.0"
                    pruned_indices_dict_bag[layer_key] = {
                        "input": pruned_indices_dict_bag[f"{layer_name}.0.conv1"][
                            "input"
                        ],
                        "output": pruned_indices_dict_bag[f"{layer_name}.0.conv2"][
                            "output"
                        ],
                    }

        # fc
        pruned_indices_dict_bag["fc"] = {"input": pruned_out_indices_bags[i]}
        pruned_indices_dict_bags.append(pruned_indices_dict_bag)

    return pruned_indices_dict_bags


def prune_linear_layer_legacy(
    linear_layer, pruned_indices: Dict[str, np.ndarray] = None
):
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
        input_indices_to_keep = np.sort(
            np.setdiff1d(
                range(input_features), pruned_indices.get("input", np.array([]))
            )
        )
        output_indices_to_keep = np.sort(
            np.setdiff1d(
                range(output_features), pruned_indices.get("output", np.array([]))
            )
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


def prune_linear_layer(
    layer, prune_indices_dict: Optional[Dict[str, np.ndarray]] = None
):
    """Prune a linear layer by using provided `prune_indices_dict` to directly select neurons to drop.

    Args:
        - linear_layer: The linear layer to prune (an instance of torch.nn.Linear).
        - prune_indices_dict: A dictionary with keys 'input' and 'output', indicating the indices of neurons to prune directly.

    Returns:
        - new_layer: The new linear layer with pruned neurons and smaller size.
    """
    assert isinstance(
        layer, torch.nn.Linear
    ), "Input layer must be an instance of torch.nn.Linear"

    in_features = layer.in_features
    out_features = layer.out_features

    if prune_indices_dict is not None:
        # Note: Make sure the pruned indices are in relative order
        input_indices_keep = np.sort(
            np.setdiff1d(
                range(in_features), prune_indices_dict.get("input", np.array([]))
            )
        )
        output_indices_keep = np.sort(
            np.setdiff1d(
                range(out_features), prune_indices_dict.get("output", np.array([]))
            )
        )

        new_weight = torch.index_select(
            torch.index_select(layer.weight.data, 0, torch.tensor(output_indices_keep)),
            1,
            torch.tensor(input_indices_keep),
        )

        new_bias = (
            torch.index_select(layer.bias.data, 0, torch.tensor(output_indices_keep))
            if layer.bias is not None
            else None
        )
    else:
        input_indices_keep = np.arange(in_features)
        output_indices_keep = np.arange(out_features)
        new_weight = layer.weight.detach().clone()
        new_bias = layer.bias.detach.clone() if layer.bias is not None else None

    new_layer = torch.nn.Linear(len(input_indices_keep), len(output_indices_keep))
    with torch.no_grad():
        new_layer.weight.data.copy_(new_weight)
        if new_bias is not None:
            new_layer.bias.data.copy_(new_bias)

    return new_layer


def prune_conv_layer_legacy(conv_layer, pruned_indices: Dict[str, np.ndarray] = None):
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
        in_indices_to_keep = np.sort(
            np.setdiff1d(range(in_channels), pruned_indices.get("input", np.array([])))
        )
        out_indices_to_keep = np.sort(
            np.setdiff1d(
                range(out_channels), pruned_indices.get("output", np.array([]))
            )
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


def prune_conv_layer(layer, prune_indices_dict: Optional[Dict[str, np.ndarray]] = None):
    """Prune a convolution layer by using provided `prune_indices_dict` to directly select channels to drop.

    Args:
        - layer: The convolution layer to prune (an instance of torch.nn.Conv2d).
        - prune_indices_dict: A dictionary with keys 'input' and 'output', indicating the indices of channels to prune directly.

    Returns:
        - new_layer: The new convolution layer with pruned channels.
    """
    assert isinstance(
        layer, torch.nn.Conv2d
    ), "Input layer must be an instance of torch.nn.Conv2d"

    in_channels = layer.in_channels
    out_channels = layer.out_channels

    if prune_indices_dict is not None:
        # Note: Make sure the pruned indices are in relative order
        in_indices_keep = np.sort(
            np.setdiff1d(
                range(in_channels), prune_indices_dict.get("input", np.array([]))
            )
        )
        out_indices_keep = np.sort(
            np.setdiff1d(
                range(out_channels), prune_indices_dict.get("output", np.array([]))
            )
        )

        new_weight = torch.index_select(
            torch.index_select(layer.weight.data, 0, torch.tensor(out_indices_keep)),
            1,
            torch.tensor(in_indices_keep),
        )

        new_bias = (
            torch.index_select(layer.bias.data, 0, torch.tensor(out_indices_keep))
            if layer.bias is not None
            else None
        )
    else:
        in_indices_keep = np.arange(in_channels)
        out_indices_keep = np.arange(out_channels)
        new_weight = layer.weight.detach().clone()
        new_bias = layer.bias.detach().clone() if layer.bias is not None else None

    new_layer = torch.nn.Conv2d(
        len(in_indices_keep),
        len(out_indices_keep),
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=(layer.bias is not None),
    )
    with torch.no_grad():
        new_layer.weight.data.copy_(new_weight)
        if new_bias is not None:
            new_layer.bias.data.copy_(new_bias)

    return new_layer


def prune_cnn_legacy(
    original_cnn: CNN, dropout_rate=0.5, scaling=True, **indices_to_prune
):
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


def prune_cnn_v2(model, dropout_rate=0.5):
    pruned_indices_dict = {}

    conv1 = model.layer1[0]
    num_output_channels_to_prune_conv1 = int(conv1.out_channels * dropout_rate)
    output_indices_to_prune_conv1 = np.random.choice(
        conv1.out_channels, num_output_channels_to_prune_conv1, replace=False
    )
    indices_to_prune_conv1 = {"output": output_indices_to_prune_conv1}
    pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)
    pruned_indices_dict["layer1"] = indices_to_prune_conv1

    conv2 = model.layer2[0]
    num_output_channels_to_prune_conv2 = int(conv2.out_channels * dropout_rate)
    output_indices_to_prune_conv2 = np.random.choice(
        conv2.out_channels, num_output_channels_to_prune_conv2, replace=False
    )
    indices_to_prune_conv2 = {
        "input": output_indices_to_prune_conv1,
        "output": output_indices_to_prune_conv2,
    }
    pruned_layer2 = prune_conv_layer(conv2, indices_to_prune_conv2)
    pruned_indices_dict["layer2"] = indices_to_prune_conv2

    conv3 = model.layer3[0]
    num_output_channels_to_prune_conv3 = int(conv3.out_channels * dropout_rate)
    output_indices_to_prune_conv3 = np.random.choice(
        conv3.out_channels, num_output_channels_to_prune_conv3, replace=False
    )
    indices_to_prune_conv3 = {
        "input": output_indices_to_prune_conv2,
        "output": output_indices_to_prune_conv3,
    }
    pruned_layer3 = prune_conv_layer(conv3, indices_to_prune_conv3)
    pruned_indices_dict["layer3"] = indices_to_prune_conv3

    fc = model.fc
    input_indices_to_prune_fc = []
    for channel_index in output_indices_to_prune_conv3:
        start_index = channel_index * 4 * 4
        end_index = (channel_index + 1) * 4 * 4
        input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
    input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
    indices_to_prune_fc = {"input": input_indices_to_prune_fc}
    pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)
    pruned_indices_dict["fc"] = indices_to_prune_fc

    pruned_cnn = CNN()
    pruned_cnn.layer1[0] = pruned_layer1
    pruned_cnn.layer2[0] = pruned_layer2
    pruned_cnn.layer3[0] = pruned_layer3
    pruned_cnn.fc = pruned_fc

    pruned_cnn.layer1.add_module("scaling", DropoutScaling(dropout_rate))
    pruned_cnn.layer2.add_module("scaling", DropoutScaling(dropout_rate))
    pruned_cnn.layer3.add_module("scaling", DropoutScaling(dropout_rate))

    return pruned_cnn, pruned_indices_dict


def prune_cnn(
    model: nn.Module,
    dropout_rate: float,
    optional_indices_dict: Optional[Dict[str, np.ndarray]] = None,
    scaling: bool = True,
):
    """Prune a CNN model by using the provided dropout rate and optional indices to prune.
    Pick the right number of channels to prune for each layer based on the dropout rate.
    Then pick the channels to prune, and the indices picked must be within the optional indices.

    Args:
        model: The CNN model to prune.
        dropout_rate: The dropout rate to use for pruning.
        optional_indices_dict: A dictionary containing the optional indices to prune for each layer.

    Returns:
        pruned_model: The pruned CNN model.
        pruned_indices_dict: A dictionary containing the indices to prune for each pruned layer.
    """
    pruned_indices_dict = {}

    conv1 = model.layer1[0]
    num_output_channels_to_prune_conv1 = int(conv1.out_channels * dropout_rate)
    if optional_indices_dict is not None:
        optional_output_indices_conv1 = optional_indices_dict["layer1"]["output"]
        assert num_output_channels_to_prune_conv1 <= len(optional_output_indices_conv1)
    else:
        optional_output_indices_conv1 = np.arange(conv1.out_channels)
    output_indices_to_prune_conv1 = np.random.choice(
        optional_output_indices_conv1, num_output_channels_to_prune_conv1, replace=False
    )
    indices_to_prune_conv1 = {"output": output_indices_to_prune_conv1}
    pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)
    pruned_indices_dict["layer1"] = indices_to_prune_conv1

    conv2 = model.layer2[0]
    num_output_channels_to_prune_conv2 = int(conv2.out_channels * dropout_rate)
    if optional_indices_dict is not None:
        optional_output_indices_conv2 = optional_indices_dict["layer2"]["output"]
        assert num_output_channels_to_prune_conv2 <= len(optional_output_indices_conv2)
    else:
        optional_output_indices_conv2 = np.arange(conv2.out_channels)
    output_indices_to_prune_conv2 = np.random.choice(
        optional_output_indices_conv2, num_output_channels_to_prune_conv2, replace=False
    )
    indices_to_prune_conv2 = {
        "input": output_indices_to_prune_conv1,
        "output": output_indices_to_prune_conv2,
    }
    pruned_layer2 = prune_conv_layer(conv2, indices_to_prune_conv2)
    pruned_indices_dict["layer2"] = indices_to_prune_conv2

    conv3 = model.layer3[0]
    num_output_channels_to_prune_conv3 = int(conv3.out_channels * dropout_rate)
    if optional_indices_dict is not None:
        optional_output_indices_conv3 = optional_indices_dict["layer3"]["output"]
        assert num_output_channels_to_prune_conv3 <= len(optional_output_indices_conv3)
    else:
        optional_output_indices_conv3 = np.arange(conv3.out_channels)
    output_indices_to_prune_conv3 = np.random.choice(
        optional_output_indices_conv3, num_output_channels_to_prune_conv3, replace=False
    )
    indices_to_prune_conv3 = {
        "input": output_indices_to_prune_conv2,
        "output": output_indices_to_prune_conv3,
    }
    pruned_layer3 = prune_conv_layer(conv3, indices_to_prune_conv3)
    pruned_indices_dict["layer3"] = indices_to_prune_conv3

    fc = model.fc
    input_indices_to_prune_fc = []
    for channel_index in output_indices_to_prune_conv3:
        start_index = channel_index * 4 * 4
        end_index = (channel_index + 1) * 4 * 4
        input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
    input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
    indices_to_prune_fc = {"input": input_indices_to_prune_fc}
    pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)
    pruned_indices_dict["fc"] = indices_to_prune_fc

    pruned_cnn = CNN()
    pruned_cnn.layer1[0] = pruned_layer1
    pruned_cnn.layer2[0] = pruned_layer2
    pruned_cnn.layer3[0] = pruned_layer3
    setattr(pruned_cnn, "fc", pruned_fc)

    if scaling:
        pruned_cnn.layer1.add_module("scaling", DropoutScaling(dropout_rate))
        pruned_cnn.layer2.add_module("scaling", DropoutScaling(dropout_rate))
        pruned_cnn.layer3.add_module("scaling", DropoutScaling(dropout_rate))

    return pruned_cnn, pruned_indices_dict


def prune_resnet18(
    model: ResNet,
    dropout_rate: float = 0.5,
    optional_indices_dict: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Prune a ResNet18 model by using the provided dropout rate and optional indices to prune.
    Pick the right number of channels to prune for each layer based on the dropout rate.
    Then pick the channels to prune, and the indices picked must be within the optional indices.

    Args:
        model: The ResNet18 model to prune.
        dropout_rate: The dropout rate to use for pruning.
        optional_indices_dict: A dictionary containing the optional indices to prune for each layer.

    Returns:
        pruned_model: The pruned ResNet18 model.
        pruned_indices_dict: A dictionary containing the indices to prune for each pruned layer.
    """
    # Note: using static layer normlization

    new_model = copy.deepcopy(model)

    pruned_indices_dicts = {}

    layer_key = "conv1"
    current_layer = new_model.conv1
    num_out_channels = current_layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    if optional_indices_dict is not None:
        optional_output_indices = optional_indices_dict[layer_key]["output"]
        assert num_out_channels_to_prune <= len(optional_output_indices)
    else:
        optional_output_indices = np.arange(num_out_channels)
    out_channel_indices_to_prune = np.random.choice(
        optional_output_indices, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dicts[layer_key] = {"output": out_channel_indices_to_prune}
    new_layer = prune_conv_layer(current_layer, pruned_indices_dicts[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model, layer_key, new_layer)

    # Update layer norm's input shape
    num_out_channels_left = new_layer[0].out_channels
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 16, 16], elementwise_affine=False
    )
    setattr(new_model, "bn1", new_layer_norm)

    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    blocks = ["0", "1"]
    convs = ["conv1", "conv2"]
    layer_norm_shapes = [
        [64, 8, 8],
        [128, 4, 4],
        [256, 2, 2],
        [512, 1, 1],
    ]

    for i, layer_name in enumerate(layer_names):
        for block in blocks:
            for conv in convs:
                in_channel_indices_to_prune = out_channel_indices_to_prune  # input indices to prune should be the same as the output indices to prune of the previous layer
                layer_key = f"{layer_name}.{block}.{conv}"
                current_layer = getattr(
                    getattr(new_model, layer_name)[int(block)], conv
                )
                num_out_channels = current_layer.out_channels
                num_out_channels_to_prune = int(num_out_channels * dropout_rate)
                if optional_indices_dict is not None:
                    optional_output_indices = optional_indices_dict[layer_key]["output"]
                    assert num_out_channels_to_prune <= len(optional_output_indices)
                else:
                    optional_output_indices = np.arange(num_out_channels)
                out_channel_indices_to_prune = np.random.choice(
                    optional_output_indices, num_out_channels_to_prune, replace=False
                )
                pruned_indices_dicts[layer_key] = {
                    "input": in_channel_indices_to_prune,
                    "output": out_channel_indices_to_prune,
                }
                new_layer = prune_conv_layer(
                    current_layer, pruned_indices_dicts[layer_key]
                )
                new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
                setattr(getattr(new_model, layer_name)[int(block)], conv, new_layer)

                # Update layer norm's input shape
                num_out_channels_left = new_layer[0].out_channels
                layer_norm_shape = layer_norm_shapes[i]
                layer_norm_shape[0] = num_out_channels_left
                new_layer_norm = nn.LayerNorm(
                    layer_norm_shape, elementwise_affine=False
                )
                setattr(
                    getattr(new_model, layer_name)[int(block)],
                    f"bn{int(conv[-1])}",
                    new_layer_norm,
                )

            # If there is downsample layer
            if (
                getattr(getattr(new_model, layer_name)[int(block)], "downsample")
                is not None
            ):
                layer_key = f"{layer_name}.{block}.downsample.0"
                current_layer = getattr(
                    getattr(new_model, layer_name)[int(block)], "downsample"
                )[0]
                pruned_indices_dicts[layer_key] = {
                    "input": pruned_indices_dicts[f"{layer_name}.{block}.conv1"][
                        "input"
                    ],
                    "output": pruned_indices_dicts[f"{layer_name}.{block}.conv2"][
                        "output"
                    ],
                }
                new_layer = prune_conv_layer(
                    current_layer, pruned_indices_dicts[layer_key]
                )
                num_out_channels_left = new_layer.out_channels
                layer_norm_shape = layer_norm_shapes[i]
                layer_norm_shape[0] = num_out_channels_left
                new_layer_norm = nn.LayerNorm(
                    layer_norm_shape, elementwise_affine=False
                )
                new_layer = nn.Sequential(
                    new_layer, DropoutScaling(dropout_rate), new_layer_norm
                )
                setattr(
                    getattr(new_model, layer_name)[int(block)], "downsample", new_layer
                )

    # ----- fc -----
    layer_key = "fc"
    current_layer = new_model.fc
    in_features_to_prune = (
        out_channel_indices_to_prune  # the last conv layer's output indices
    )
    # Since the last conv output H, W is 1, 1, we can just use the channel indices
    pruned_indices_dicts[layer_key] = {"input": in_features_to_prune}
    new_layer = prune_linear_layer(current_layer, pruned_indices_dicts[layer_key])
    setattr(new_model, layer_key, new_layer)

    return new_model, pruned_indices_dicts


def prune_shallow_resnet(model: ResNet, dropout_rate=0.5):
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
    new_model = copy.deepcopy(model)

    pruned_indices_dicts = {}

    layer_key = "conv1"
    current_layer = new_model.conv1
    num_out_channels = current_layer.out_channels
    num_out_channels_to_prune = int(num_out_channels * dropout_rate)
    out_channel_indices_to_prune = np.random.choice(
        num_out_channels, num_out_channels_to_prune, replace=False
    )
    pruned_indices_dicts[layer_key] = {"output": out_channel_indices_to_prune}
    new_layer = prune_conv_layer(current_layer, pruned_indices_dicts[layer_key])
    new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
    setattr(new_model, layer_key, new_layer)

    # Update layer norm's input shape
    num_out_channels_left = new_layer[0].out_channels
    new_layer_norm = nn.LayerNorm(
        [num_out_channels_left, 16, 16], elementwise_affine=False
    )
    setattr(new_model, "bn1", new_layer_norm)

    layer_names = ["layer1", "layer2"]
    # layer_names = ["layer1"]
    # blocks = ["0", "1"]
    blocks = ["0"]
    convs = ["conv1", "conv2"]
    layer_norm_shapes = [
        [64, 8, 8],
        [128, 4, 4],
        [256, 2, 2],
        [512, 1, 1],
    ]

    for i, layer_name in enumerate(layer_names):
        for block in blocks:
            for conv in convs:
                layer_key = f"{layer_name}.{block}.{conv}"
                current_layer = getattr(
                    getattr(new_model, layer_name)[int(block)], conv
                )
                num_out_channels = current_layer.out_channels
                num_out_channels_to_prune = int(num_out_channels * dropout_rate)
                in_channel_indices_to_prune = out_channel_indices_to_prune  # input indices to prune should be the same as the output indices to prune of the previous layer
                out_channel_indices_to_prune = np.random.choice(
                    num_out_channels, num_out_channels_to_prune, replace=False
                )
                pruned_indices_dicts[layer_key] = {
                    "input": in_channel_indices_to_prune,
                    "output": out_channel_indices_to_prune,
                }
                new_layer = prune_conv_layer(
                    current_layer, pruned_indices_dicts[layer_key]
                )
                new_layer = nn.Sequential(new_layer, DropoutScaling(dropout_rate))
                setattr(getattr(new_model, layer_name)[int(block)], conv, new_layer)

                # Update layer norm's input shape
                num_out_channels_left = new_layer[0].out_channels
                layer_norm_shape = layer_norm_shapes[i]
                layer_norm_shape[0] = num_out_channels_left
                new_layer_norm = nn.LayerNorm(
                    layer_norm_shape, elementwise_affine=False
                )
                setattr(
                    getattr(new_model, layer_name)[int(block)],
                    f"bn{int(conv[-1])}",
                    new_layer_norm,
                )

            # If there is downsample layer
            if (
                getattr(getattr(new_model, layer_name)[int(block)], "downsample")
                is not None
            ):
                layer_key = f"{layer_name}.{block}.downsample.0"
                current_layer = getattr(
                    getattr(new_model, layer_name)[int(block)], "downsample"
                )[0]
                pruned_indices_dicts[layer_key] = {
                    "input": pruned_indices_dicts[f"{layer_name}.{block}.conv1"][
                        "input"
                    ],
                    "output": pruned_indices_dicts[f"{layer_name}.{block}.conv2"][
                        "output"
                    ],
                }
                new_layer = prune_conv_layer(
                    current_layer, pruned_indices_dicts[layer_key]
                )
                num_out_channels_left = new_layer.out_channels
                layer_norm_shape = layer_norm_shapes[i]
                layer_norm_shape[0] = num_out_channels_left
                new_layer_norm = nn.LayerNorm(
                    layer_norm_shape, elementwise_affine=False
                )
                new_layer = nn.Sequential(
                    new_layer, DropoutScaling(dropout_rate), new_layer_norm
                )
                setattr(
                    getattr(new_model, layer_name)[int(block)], "downsample", new_layer
                )

    # ----- fc -----
    layer_key = "fc"
    current_layer = new_model.fc
    in_features_to_prune = (
        out_channel_indices_to_prune  # the last conv layer's output indices
    )
    # in_features_to_prune = []
    # for channel_index in out_channel_indices_to_prune:
    #     start_index = channel_index * 4 * 4
    #     end_index = (channel_index + 1) * 4 * 4
    #     in_features_to_prune.extend(list(range(start_index, end_index)))
    # in_features_to_prune = np.sort(in_features_to_prune)
    # Since the last conv output H, W is 1, 1, we can just use the channel indices
    pruned_indices_dicts[layer_key] = {"input": in_features_to_prune}
    new_layer = prune_linear_layer(current_layer, pruned_indices_dicts[layer_key])
    setattr(new_model, layer_key, new_layer)

    return new_model, pruned_indices_dicts


def prune_cnn_groups_nested(model: nn.Module, dropout_rates: List[float]):
    """Prune a group of CNNs with different dropout rates.
    The `dropout_rates` list should contain the dropout rates in descending order.
    Prune the model with the first dropout rate and get a `pruned_indices_dict`,
    then prune the model with the second dropout rate but within the first pruned indices,
    and so on.

    Args:
        model: The CNN model to prune.
        dropout_rates: A list of dropout rates in descending order.

    Returns:
        pruned_models: A list of pruned CNN models.
        pruned_indices_dicts: A list of dictionaries containing the indices to prune for each pruned layer.
    """
    # Make sure the dropout rates in descending order
    dropout_rates = sorted(dropout_rates, reverse=True)

    pruned_models = []
    pruned_indices_dicts = []

    for i, dropout_rate in enumerate(dropout_rates):
        if i == 0:
            pruned_model, pruned_indices_dict = prune_cnn(model, dropout_rate)
            pruned_models.append(pruned_model)
            pruned_indices_dicts.append(pruned_indices_dict)
        else:
            pruned_model, pruned_indices_dict = prune_cnn(
                model, dropout_rate, pruned_indices_dicts[i - 1]
            )
            pruned_models.append(pruned_model)
            pruned_indices_dicts.append(pruned_indices_dict)

    return pruned_models, pruned_indices_dicts


def prune_resnet18_groups_nested(model: ResNet, dropout_rates: List[float]):
    """Prune a group of ResNet models with different dropout rates.
    The `dropout_rates` list should contain the dropout rates in descending order.
    Prune the model with the first dropout rate and get a `pruned_indices_dict`,
    then prune the model with the second dropout rate but within the first pruned indices,
    and so on.

    Args:
        model: The ResNet model to prune.
        dropout_rates: A list of dropout rates in descending order.

    Returns:
        pruned_models: A list of pruned ResNet models.
        pruned_indices_dicts: A list of dictionaries containing the indices to prune for each pruned layer.
    """
    # Make sure the dropout rates in descending order
    dropout_rates = sorted(dropout_rates, reverse=True)

    pruned_models = []
    pruned_indices_dicts = []

    for i, dropout_rate in enumerate(dropout_rates):
        if i == 0:
            pruned_model, pruned_indices_dict = prune_resnet18(model, dropout_rate)
            pruned_models.append(pruned_model)
            pruned_indices_dicts.append(pruned_indices_dict)
        else:
            pruned_model, pruned_indices_dict = prune_resnet18(
                model, dropout_rate, pruned_indices_dicts[i - 1]
            )
            pruned_models.append(pruned_model)
            pruned_indices_dicts.append(pruned_indices_dict)

    return pruned_models, pruned_indices_dicts
