import numpy as np
import torch
import torch.nn as nn

from modules.models import CNN, DropoutScaling
from modules.utils import aggregate_conv_layers, aggregate_linear_layers


def prune_conv_layer(conv_layer, p, prune_input=True, prune_output=True):
    """
    Prune a Conv2d layer based on the given proportion p and pruning flags for input and output channels.

    Parameters:
    - conv_layer: nn.Conv2d, the original convolutional layer to be pruned.
    - p: float, the proportion of channels to be pruned.
    - prune_input: bool, whether to prune the input channels.
    - prune_output: bool, whether to prune the output channels.

    Returns:
    - new_conv_layer: nn.Conv2d, the new convolutional layer after pruning.
    """
    # Calculate the new number of in_channels and out_channels based on pruning flags
    original_in_channels = conv_layer.in_channels
    original_out_channels = conv_layer.out_channels
    new_in_channels = (
        int(original_in_channels * (1 - p)) if prune_input else original_in_channels
    )
    new_out_channels = (
        int(original_out_channels * (1 - p)) if prune_output else original_out_channels
    )

    # Ensure that the new channels are at least 1
    new_in_channels = max(1, new_in_channels)
    new_out_channels = max(1, new_out_channels)

    # Extract the original weights and bias
    original_weights = conv_layer.weight.data
    original_bias = conv_layer.bias.data if conv_layer.bias is not None else None

    # Extract the corresponding part of the weights and bias based on pruning flags
    if prune_input and prune_output:
        new_weights = original_weights[:new_out_channels, :new_in_channels, :, :]
    elif prune_input:
        new_weights = original_weights[:, :new_in_channels, :, :]
    elif prune_output:
        new_weights = original_weights[:new_out_channels, :, :, :]
    else:
        new_weights = original_weights

    new_bias = (
        original_bias[:new_out_channels]
        if original_bias is not None and prune_output
        else original_bias
    )

    # Create a new Conv2d layer with the pruned weights and bias
    new_conv_layer = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=new_out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=conv_layer.bias is not None,
    )

    # Assign the pruned weights and bias to the new Conv2d layer
    new_conv_layer.weight.data = new_weights
    if new_bias is not None:
        new_conv_layer.bias.data = new_bias

    return new_conv_layer


def expand_pruned_conv_layer(pruned_conv_layer, original_conv_layer):
    """
    Expand a pruned convolutional layer to match the shape of the original convolutional layer
    by copying the weights and biases.

    Parameters:
    - pruned_conv_layer: nn.Conv2d, the pruned convolutional layer.
    - original_conv_layer: nn.Conv2d, the original convolutional layer before pruning.

    Returns:
    - expanded_conv_layer: nn.Conv2d, the expanded convolutional layer.
    """
    # Step 1: Get weights and biases from both layers
    pruned_weights = pruned_conv_layer.weight.data
    pruned_bias = (
        pruned_conv_layer.bias.data if pruned_conv_layer.bias is not None else None
    )

    original_weights_shape = original_conv_layer.weight.shape
    original_bias_shape = (
        original_conv_layer.bias.shape if original_conv_layer.bias is not None else None
    )

    # Step 2: Create new tensors for weights and biases to match the original layer's shape
    expanded_weights = torch.zeros(original_weights_shape)
    expanded_bias = (
        torch.zeros(original_bias_shape) if original_bias_shape is not None else None
    )

    # Step 3: Copy pruned weights and biases into the new tensors
    for i in range(original_weights_shape[0]):
        for j in range(original_weights_shape[1]):
            expanded_weights[i, j, :, :] = pruned_weights[
                i % pruned_weights.shape[0], j % pruned_weights.shape[1], :, :
            ]

    if expanded_bias is not None:
        for i in range(original_bias_shape[0]):
            expanded_bias[i] = pruned_bias[i % pruned_bias.shape[0]]

    # Step 4: Create a new Conv2d layer with the expanded weights and bias
    expanded_conv_layer = nn.Conv2d(
        in_channels=original_conv_layer.in_channels,
        out_channels=original_conv_layer.out_channels,
        kernel_size=original_conv_layer.kernel_size,
        stride=original_conv_layer.stride,
        padding=original_conv_layer.padding,
        dilation=original_conv_layer.dilation,
        groups=original_conv_layer.groups,
        bias=original_conv_layer.bias is not None,
    )

    # Assign the expanded weights and bias to the new Conv2d layer
    expanded_conv_layer.weight.data = expanded_weights
    if expanded_bias is not None:
        expanded_conv_layer.bias.data = expanded_bias

    return expanded_conv_layer


def prune_linear_layer(
    linear_layer,
    p,
    prune_input=True,
    prune_output=True,
    new_in_features=None,
    new_out_features=None,
):
    """
    Prune a Linear layer based on the given proportion p and pruning flags for input and output features.

    Parameters:
    - linear_layer: nn.Linear, the original linear layer to be pruned.
    - p: float, the proportion of features to be pruned.
    - prune_input: bool, whether to prune the input features.
    - prune_output: bool, whether to prune the output features.

    Returns:
    - new_linear_layer: nn.Linear, the new linear layer after pruning.
    """
    # Calculate the new number of input and output features based on pruning flags
    original_in_features = linear_layer.in_features
    original_out_features = linear_layer.out_features
    if new_in_features is None:
        new_in_features = (
            int(original_in_features * (1 - p)) if prune_input else original_in_features
        )
        new_in_features = max(1, new_in_features)
    if new_out_features is None:
        new_out_features = (
            int(original_out_features * (1 - p))
            if prune_output
            else original_out_features
        )
        new_out_features = max(1, new_out_features)

    # Extract the original weights and bias
    original_weights = linear_layer.weight.data
    original_bias = linear_layer.bias.data if linear_layer.bias is not None else None

    # Create new weights and bias based on the pruned size
    new_weights = original_weights[:new_out_features, :new_in_features]
    new_bias = original_bias[:new_out_features] if original_bias is not None else None

    # Create a new Linear layer with the pruned weights and bias
    new_linear_layer = nn.Linear(new_in_features, new_out_features)
    new_linear_layer.weight.data = new_weights
    if new_bias is not None:
        new_linear_layer.bias.data = new_bias

    return new_linear_layer


def expand_pruned_linear_layer(pruned_linear_layer, original_linear_layer):
    """
    Expands a pruned linear layer's weights and biases to match the shape of an original linear layer.

    Parameters:
    - pruned_linear_layer: nn.Linear, the pruned linear layer with smaller weights and biases.
    - original_linear_layer: nn.Linear, the original linear layer with the target shape for weights and biases.

    Returns:
    - expanded_linear_layer: nn.Linear, the expanded linear layer with weights and biases matching the original layer.
    """
    # Step 1: Get weights and biases from both layers
    pruned_weights = pruned_linear_layer.weight.data
    pruned_bias = (
        pruned_linear_layer.bias.data if pruned_linear_layer.bias is not None else None
    )

    original_weights_shape = original_linear_layer.weight.shape
    original_bias_shape = (
        original_linear_layer.bias.shape
        if original_linear_layer.bias is not None
        else None
    )

    # Step 2: Create new tensors for weights and biases to match the original layer's shape
    expanded_weights = torch.zeros(original_weights_shape)
    expanded_bias = (
        torch.zeros(original_bias_shape) if original_bias_shape is not None else None
    )

    # Step 3: Copy pruned weights and biases into the new tensors
    for i in range(original_weights_shape[0]):
        for j in range(original_weights_shape[1]):
            expanded_weights[i, j] = pruned_weights[
                i % pruned_weights.shape[0], j % pruned_weights.shape[1]
            ]

    if expanded_bias is not None:
        for i in range(original_bias_shape[0]):
            expanded_bias[i] = pruned_bias[i % pruned_bias.shape[0]]

    # Step 4: Create a new Linear layer with the expanded weights and bias
    expanded_linear_layer = nn.Linear(
        original_linear_layer.in_features,
        original_linear_layer.out_features,
        bias=original_linear_layer.bias is not None,
    )
    expanded_linear_layer.weight.data = expanded_weights
    if expanded_bias is not None:
        expanded_linear_layer.bias.data = expanded_bias

    return expanded_linear_layer


def prune_cnn(original_cnn, p, scaling=True):
    pruned_cnn = CNN()

    pruned_conv_layer1 = prune_conv_layer(
        original_cnn.layer1[0], p, prune_input=False, prune_output=True
    )
    pruned_conv_layer2 = prune_conv_layer(
        original_cnn.layer2[0], p, prune_input=True, prune_output=True
    )
    pruned_conv_layer3 = prune_conv_layer(
        original_cnn.layer3[0], p, prune_input=True, prune_output=True
    )
    # pruned_fc_layer = prune_linear_layer(
    #     original_cnn.fc, p, prune_input=True, prune_output=False
    # )
    num_keep_features = pruned_conv_layer3.out_channels * 4 * 4
    pruned_fc_layer = prune_linear_layer(
        original_cnn.fc,
        p,
        prune_input=True,
        prune_output=False,
        new_in_features=num_keep_features,
    )

    pruned_cnn.layer1[0] = pruned_conv_layer1
    # pruned_cnn.layer1[1] = nn.BatchNorm2d(pruned_conv_layer1.out_channels)
    # pruned_cnn.layer1.add_module("dropout", DropoutScaling(p))
    pruned_cnn.layer2[0] = pruned_conv_layer2
    # pruned_cnn.layer2[1] = nn.BatchNorm2d(pruned_conv_layer2.out_channels)
    # pruned_cnn.layer2.add_module("dropout", DropoutScaling(p))
    pruned_cnn.layer3[0] = pruned_conv_layer3
    # pruned_cnn.layer3[1] = nn.BatchNorm2d(pruned_conv_layer3.out_channels)
    # pruned_cnn.layer3.add_module("dropout", DropoutScaling(p))
    pruned_cnn.fc = pruned_fc_layer

    if scaling:
        pruned_cnn.layer1.add_module("scaling", DropoutScaling(p))
        pruned_cnn.layer2.add_module("scaling", DropoutScaling(p))
        pruned_cnn.layer3.add_module("scaling", DropoutScaling(p))

    return pruned_cnn


def expand_cnn(pruned_cnn, original_cnn):
    expanded_cnn = CNN()

    expanded_conv_layer1 = expand_pruned_conv_layer(
        pruned_cnn.layer1[0], original_cnn.layer1[0]
    )
    expanded_conv_layer2 = expand_pruned_conv_layer(
        pruned_cnn.layer2[0], original_cnn.layer2[0]
    )
    expanded_conv_layer3 = expand_pruned_conv_layer(
        pruned_cnn.layer3[0], original_cnn.layer3[0]
    )
    expanded_fc_layer = expand_pruned_linear_layer(pruned_cnn.fc, original_cnn.fc)

    expanded_cnn.layer1[0] = expanded_conv_layer1
    expanded_cnn.layer2[0] = expanded_conv_layer2
    expanded_cnn.layer3[0] = expanded_conv_layer3
    expanded_cnn.fc = expanded_fc_layer

    return expanded_cnn


def heterofl_aggregate(original_cnn, pruned_cnns, weights):
    indices_to_prune_conv1 = []
    indices_to_prune_conv2 = []
    indices_to_prune_conv3 = []
    indices_to_prune_fc = []

    for i, cnn in enumerate(pruned_cnns):
        indices_to_prune_conv1.append(
            {
                "output": np.setdiff1d(
                    range(original_cnn.layer1[0].out_channels),
                    range(cnn.layer1[0].out_channels),
                ),
            }
        )
        indices_to_prune_conv2.append(
            {
                "input": np.setdiff1d(
                    range(original_cnn.layer2[0].in_channels),
                    range(cnn.layer2[0].in_channels),
                ),
                "output": np.setdiff1d(
                    range(original_cnn.layer2[0].out_channels),
                    range(cnn.layer2[0].out_channels),
                ),
            }
        )
        indices_to_prune_conv3.append(
            {
                "input": np.setdiff1d(
                    range(original_cnn.layer3[0].in_channels),
                    range(cnn.layer3[0].in_channels),
                ),
                "output": np.setdiff1d(
                    range(original_cnn.layer3[0].out_channels),
                    range(cnn.layer3[0].out_channels),
                ),
            }
        )
        indices_to_prune_fc.append(
            {
                "input": np.setdiff1d(
                    range(original_cnn.fc.in_features), range(cnn.fc.in_features)
                ),
            }
        )

    aggregate_conv_layers(
        original_cnn.layer1[0],
        [cnn.layer1[0] for cnn in pruned_cnns],
        indices_to_prune_conv1,
        weights,
    )
    aggregate_conv_layers(
        original_cnn.layer2[0],
        [cnn.layer2[0] for cnn in pruned_cnns],
        indices_to_prune_conv2,
        weights,
    )
    aggregate_conv_layers(
        original_cnn.layer3[0],
        [cnn.layer3[0] for cnn in pruned_cnns],
        indices_to_prune_conv3,
        weights,
    )
    aggregate_linear_layers(
        original_cnn.fc,
        [cnn.fc for cnn in pruned_cnns],
        indices_to_prune_fc,
        weights,
    )


def vanilla_aggregate(global_cnn, client_cnns, weights):
    indices_to_prune_conv1 = []
    indices_to_prune_conv2 = []
    indices_to_prune_conv3 = []
    indices_to_prune_fc = []

    for i, cnn in enumerate(client_cnns):
        indices_to_prune_conv1.append({})
        indices_to_prune_conv2.append({})
        indices_to_prune_conv3.append({})
        indices_to_prune_fc.append({})

    aggregate_conv_layers(
        global_cnn.layer1[0],
        [cnn.layer1[0] for cnn in client_cnns],
        indices_to_prune_conv1,
        weights,
    )
    aggregate_conv_layers(
        global_cnn.layer2[0],
        [cnn.layer2[0] for cnn in client_cnns],
        indices_to_prune_conv2,
        weights,
    )
    aggregate_conv_layers(
        global_cnn.layer3[0],
        [cnn.layer3[0] for cnn in client_cnns],
        indices_to_prune_conv3,
        weights,
    )
    aggregate_linear_layers(
        global_cnn.fc,
        [cnn.fc for cnn in client_cnns],
        indices_to_prune_fc,
        weights,
    )
