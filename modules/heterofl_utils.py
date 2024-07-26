import torch
import torch.nn as nn

from modules.aggregation import aggregate_conv_layers, aggregate_linear_layers
from modules.models import CNN, DropoutScaling


def prune_conv_layer(conv_layer, p, position, prune_input=True, prune_output=True):
    """
    Prune a Conv2d layer based on the given proportion p and position index for preserving weights,
    and pruning flags for input and output channels. Additionally, return the indices of pruned input and output channels.

    Parameters:
    - conv_layer: nn.Conv2d, the original convolutional layer to be pruned.
    - p: float, the proportion of channels to be pruned.
    - position: int, index of the position of the weights to preserve based on 1/(1-p) possible positions.
    - prune_input: bool, whether to prune the input channels.
    - prune_output: bool, whether to prune the output channels.

    Returns:
    - new_conv_layer: nn.Conv2d, the new convolutional layer after pruning.
    - pruned_input_indices: list, indices of pruned input channels.
    - pruned_output_indices: list, indices of pruned output channels.
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

    # TODO: Consider when 1 / (1 - p) is not an integer
    num_positions = int(1 / (1 - p))

    if position < 0 or position >= num_positions:
        raise ValueError(
            f"Invalid position index. Choose an integer between 0 and {num_positions - 1}."
        )

    # Calculate the segment size for each position
    segment_size_in = original_in_channels // num_positions
    segment_size_out = original_out_channels // num_positions

    # Calculate start and end indices for input and output channels
    start_in = position * segment_size_in
    end_in = start_in + new_in_channels
    start_out = position * segment_size_out
    end_out = start_out + new_out_channels

    # Adjust for boundary conditions
    if end_in > original_in_channels:
        end_in = original_in_channels
        start_in = end_in - new_in_channels
    if end_out > original_out_channels:
        end_out = original_out_channels
        start_out = end_out - new_out_channels

    # Extract the original weights and bias
    original_weights = conv_layer.weight.data
    original_bias = conv_layer.bias.data if conv_layer.bias is not None else None

    # # Extract the corresponding part of the weights and bias based on pruning flags
    # if prune_input and prune_output:
    #     new_weights = original_weights[:new_out_channels, :new_in_channels, :, :]
    # elif prune_input:
    #     new_weights = original_weights[:, :new_in_channels, :, :]
    # elif prune_output:
    #     new_weights = original_weights[:new_out_channels, :, :, :]
    # else:
    #     new_weights = original_weights
    new_weights = original_weights[start_out:end_out, start_in:end_in, :, :]

    # new_bias = (
    #     original_bias[:new_out_channels]
    #     if original_bias is not None and prune_output
    #     else original_bias
    # )
    new_bias = (
        original_bias[start_out:end_out]
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

    # Calculate preserved indices
    preserved_input_indices = list(range(start_in, end_in))
    preserved_output_indices = list(range(start_out, end_out))

    # Calculate pruned indices
    all_input_indices = set(range(original_in_channels))
    all_output_indices = set(range(original_out_channels))
    pruned_input_indices = list(all_input_indices - set(preserved_input_indices))
    pruned_output_indices = list(all_output_indices - set(preserved_output_indices))

    return new_conv_layer, pruned_input_indices, pruned_output_indices


def prune_linear_layer(
    linear_layer,
    p,
    position="left",
    prune_input=True,
    prune_output=True,
    new_in_features=None,
    new_out_features=None,
):
    """
    Prune a Linear layer based on the given proportion p and pruning flags for input and output features.
    Additionally, return the indices of pruned input and output features.

    Parameters:
    - linear_layer: nn.Linear, the original linear layer to be pruned.
    - p: float, the proportion of features to be pruned.
    - prune_input: bool, whether to prune the input features.
    - prune_output: bool, whether to prune the output features.

    Returns:
    - new_linear_layer: nn.Linear, the new linear layer after pruning.
    - pruned_input_indices: list, indices of pruned input features.
    - pruned_output_indices: list, indices of pruned output features.
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

    # Adjust indices based on the position parameter
    if position == "center":
        start_in = (original_in_features - new_in_features) // 2
        end_in = start_in + new_in_features
        start_out = (original_out_features - new_out_features) // 2
        end_out = start_out + new_out_features
    elif position == "left":
        start_in = 0
        end_in = new_in_features
        start_out = 0
        end_out = new_out_features
    elif position == "right":
        start_in = original_in_features - new_in_features
        end_in = original_in_features
        start_out = original_out_features - new_out_features
        end_out = original_out_features
    else:
        raise ValueError("Invalid position value. Choose 'left', 'center', or 'right'.")

    # Extract the original weights and bias
    original_weights = linear_layer.weight.data
    original_bias = linear_layer.bias.data if linear_layer.bias is not None else None

    # Create new weights and bias based on the pruned size
    # new_weights = original_weights[:new_out_features, :new_in_features]
    # new_bias = original_bias[:new_out_features] if original_bias is not None else None
    new_weights = original_weights[start_out:end_out, start_in:end_in]
    new_bias = original_bias[start_out:end_out] if original_bias is not None else None

    # Create a new Linear layer with the pruned weights and bias
    new_linear_layer = nn.Linear(new_in_features, new_out_features)
    new_linear_layer.weight.data = new_weights
    if new_bias is not None:
        new_linear_layer.bias.data = new_bias

    # Calculate preserved indices
    preserved_input_indices = (
        list(range(start_in, end_in))
        if prune_input
        else list(range(original_in_features))
    )
    preserved_output_indices = (
        list(range(start_out, end_out))
        if prune_output
        else list(range(original_out_features))
    )

    # Calculate pruned indices
    all_input_indices = set(range(original_in_features))
    all_output_indices = set(range(original_out_features))
    pruned_input_indices = list(all_input_indices - set(preserved_input_indices))
    pruned_output_indices = list(all_output_indices - set(preserved_output_indices))

    return new_linear_layer, pruned_input_indices, pruned_output_indices


def prune_linear_layer_by_indices(
    linear_layer,
    start_prune_index_input,
    end_prune_index_input,
    start_prune_index_output,
    end_prune_index_output,
):
    """
    Prune a Linear layer based on the specified start and end indices for input and output features to be pruned.
    Returns a new linear layer with the specified features removed.

    Parameters:
    - linear_layer: nn.Linear, the original linear layer to be pruned.
    - start_prune_index_input: int, the start index for input features to be pruned.
    - end_prune_index_input: int, the end index (exclusive) for input features to be pruned.
    - start_prune_index_output: int, the start index for output features to be pruned.
    - end_prune_index_output: int, the end index (exclusive) for output features to be pruned.

    Returns:
    - new_linear_layer: nn.Linear, the new linear layer after pruning.
    """

    # Validate indices
    assert (
        0 <= start_prune_index_input < end_prune_index_input <= linear_layer.in_features
    ), "Invalid input indices."
    assert (
        0
        <= start_prune_index_output
        < end_prune_index_output
        <= linear_layer.out_features
    ), "Invalid output indices."

    # Calculate new in_features and out_features after pruning
    new_in_features = linear_layer.in_features - (
        end_prune_index_input - start_prune_index_input
    )
    new_out_features = linear_layer.out_features - (
        end_prune_index_output - start_prune_index_output
    )

    # Extract the original weights and bias
    original_weights = linear_layer.weight.data
    original_bias = linear_layer.bias.data if linear_layer.bias is not None else None

    # Create masks for input and output features to keep
    input_mask = [
        i
        for i in range(linear_layer.in_features)
        if not (start_prune_index_input <= i < end_prune_index_input)
    ]
    output_mask = [
        i
        for i in range(linear_layer.out_features)
        if not (start_prune_index_output <= i < end_prune_index_output)
    ]

    # Apply masks to create new weights and bias
    new_weights = original_weights[output_mask, :][:, input_mask]
    new_bias = original_bias[output_mask] if original_bias is not None else None

    # Create a new Linear layer with the pruned weights and bias
    new_linear_layer = nn.Linear(
        new_in_features, new_out_features, bias=linear_layer.bias is not None
    )
    new_linear_layer.weight.data = new_weights
    if new_bias is not None:
        new_linear_layer.bias.data = new_bias

    return new_linear_layer


def retain_linear_layer_by_indices(
    linear_layer,
    retain_start_index_input,
    retain_end_index_input,
    retain_start_index_output,
    retain_end_index_output,
):
    """
    Retain a Linear layer based on the specified start and end indices for input and output features to be retained.
    Returns a new linear layer with only the specified features retained.

    Parameters:
    - linear_layer: nn.Linear, the original linear layer.
    - retain_start_index_input: int, the start index for input features to be retained.
    - retain_end_index_input: int, the end index (exclusive) for input features to be retained.
    - retain_start_index_output: int, the start index for output features to be retained.
    - retain_end_index_output: int, the end index (exclusive) for output features to be retained.

    Returns:
    - new_linear_layer: nn.Linear, the new linear layer after retaining the specified features.
    """

    # Validate indices
    assert (
        0
        <= retain_start_index_input
        < retain_end_index_input
        <= linear_layer.in_features
    ), "Invalid input indices."
    assert (
        0
        <= retain_start_index_output
        < retain_end_index_output
        <= linear_layer.out_features
    ), "Invalid output indices."

    # Calculate new in_features and out_features after retaining
    new_in_features = retain_end_index_input - retain_start_index_input
    new_out_features = retain_end_index_output - retain_start_index_output

    # Extract the original weights and bias
    original_weights = linear_layer.weight.data
    original_bias = linear_layer.bias.data if linear_layer.bias is not None else None

    # Apply indices to create new weights and bias
    new_weights = original_weights[
        retain_start_index_output:retain_end_index_output, :
    ][:, retain_start_index_input:retain_end_index_input]
    new_bias = (
        original_bias[retain_start_index_output:retain_end_index_output]
        if original_bias is not None
        else None
    )

    # Create a new Linear layer with the retained weights and bias
    new_linear_layer = nn.Linear(
        new_in_features, new_out_features, bias=linear_layer.bias is not None
    )
    new_linear_layer.weight.data = new_weights
    if new_bias is not None:
        new_linear_layer.bias.data = new_bias

    return new_linear_layer


def prune_cnn(original_cnn, p, position, scaling=True):
    # TODO: Consider when 1 / (1 - p) is not an integer
    num_positions = int(1 / (1 - p))

    if position < 0 or position >= num_positions:
        raise ValueError(
            f"Invalid position index. Choose an integer between 0 and {num_positions - 1}."
        )

    pruned_cnn = CNN()
    pruned_indices = {}

    (
        pruned_conv_layer1,
        pruned_input_indices_conv_layer1,
        pruned_output_indices_conv_layer1,
    ) = prune_conv_layer(
        original_cnn.layer1[0],
        p,
        prune_input=False,
        prune_output=True,
        position=position,
    )
    pruned_indices["layer1"] = {
        "input": pruned_input_indices_conv_layer1,
        "output": pruned_output_indices_conv_layer1,
    }

    (
        pruned_conv_layer2,
        pruned_input_indices_conv_layer2,
        pruned_output_indices_conv_layer2,
    ) = prune_conv_layer(
        original_cnn.layer2[0],
        p,
        prune_input=True,
        prune_output=True,
        position=position,
    )
    pruned_indices["layer2"] = {
        "input": pruned_input_indices_conv_layer2,
        "output": pruned_output_indices_conv_layer2,
    }

    (
        pruned_conv_layer3,
        pruned_input_indices_conv_layer3,
        pruned_output_indices_conv_layer3,
    ) = prune_conv_layer(
        original_cnn.layer3[0],
        p,
        prune_input=True,
        prune_output=True,
        position=position,
    )
    pruned_indices["layer3"] = {
        "input": pruned_input_indices_conv_layer3,
        "output": pruned_output_indices_conv_layer3,
    }

    # Calculate the preserved output indices for the conv_layer3
    all_output_indices_conv_layer3 = set(range(original_cnn.layer3[0].out_channels))
    preserved_output_indices_conv_layer3 = list(
        all_output_indices_conv_layer3 - set(pruned_output_indices_conv_layer3)
    )
    # Sort the preserved output indices
    preserved_output_indices_conv_layer3.sort()

    retain_start_index_input = preserved_output_indices_conv_layer3[0] * 4 * 4
    retain_end_index_input = (preserved_output_indices_conv_layer3[-1] + 1) * 4 * 4
    # Keep all the output features
    retain_start_index_output = 0
    retain_end_index_output = original_cnn.fc.out_features
    pruned_fc_layer = retain_linear_layer_by_indices(
        original_cnn.fc,
        retain_start_index_input,
        retain_end_index_input,
        retain_start_index_output,
        retain_end_index_output,
    )

    preserved_input_indices_fc_layer = list(
        range(retain_start_index_input, retain_end_index_input)
    )
    pruned_input_indices_fc_layer = list(
        set(range(original_cnn.fc.in_features)) - set(preserved_input_indices_fc_layer)
    )
    pruned_output_indices_fc_layer = []

    pruned_indices["fc"] = {
        "input": pruned_input_indices_fc_layer,
        "output": pruned_output_indices_fc_layer,
    }

    pruned_cnn.layer1[0] = pruned_conv_layer1
    pruned_cnn.layer2[0] = pruned_conv_layer2
    pruned_cnn.layer3[0] = pruned_conv_layer3
    pruned_cnn.fc = pruned_fc_layer

    if scaling:
        pruned_cnn.layer1.add_module("scaling", DropoutScaling(p))
        pruned_cnn.layer2.add_module("scaling", DropoutScaling(p))
        pruned_cnn.layer3.add_module("scaling", DropoutScaling(p))

    return pruned_cnn, pruned_indices


def prune_cnn_group(original_cnn, p, scaling=True):
    num_cnns = int(1 / (1 - p))

    pruned_cnns = []
    pruned_indices_group = []
    for i in range(num_cnns):
        pruned_cnn, pruned_indices = prune_cnn(
            original_cnn, p, position=i, scaling=scaling
        )
        pruned_cnns.append(pruned_cnn)
        pruned_indices_group.append(pruned_indices)

    return pruned_cnns, pruned_indices_group


def empty_pruned_indices():
    return {
        "layer1": {"input": [], "output": []},
        "layer2": {"input": [], "output": []},
        "layer3": {"input": [], "output": []},
        "fc": {"input": [], "output": []},
    }


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


def heterofl_aggregate(original_cnn, pruned_cnns, pruned_indices_list, weights):
    # indices_to_prune_conv1 = []
    # indices_to_prune_conv2 = []
    # indices_to_prune_conv3 = []
    # indices_to_prune_fc = []

    indices_to_prune_conv1_list = []
    indices_to_prune_conv2_list = []
    indices_to_prune_conv3_list = []
    indices_to_prune_fc_list = []

    # Assume pruned_indices_list is a list of dictionaries, each corresponding to pruned indices of a pruned_cnn in pruned_cnns

    for i, cnn in enumerate(pruned_cnns):
        pruned_indices = pruned_indices_list[
            i
        ]  # Get the pruned indices for the current pruned CNN

        # Directly use pruned indices from the provided dictionary
        indices_to_prune_conv1_list.append(
            {
                "output": pruned_indices["layer1"]["output"],
            }
        )
        indices_to_prune_conv2_list.append(
            {
                "input": pruned_indices["layer2"]["input"],
                "output": pruned_indices["layer2"]["output"],
            }
        )
        indices_to_prune_conv3_list.append(
            {
                "input": pruned_indices["layer3"]["input"],
                "output": pruned_indices["layer3"]["output"],
            }
        )
        indices_to_prune_fc_list.append(
            {
                "input": pruned_indices["fc"]["input"],
            }
        )

        # indices_to_prune_conv1.append(
        #     {
        #         "output": np.setdiff1d(
        #             range(original_cnn.layer1[0].out_channels),
        #             range(cnn.layer1[0].out_channels),
        #         ),
        #     }
        # )
        # indices_to_prune_conv2.append(
        #     {
        #         "input": np.setdiff1d(
        #             range(original_cnn.layer2[0].in_channels),
        #             range(cnn.layer2[0].in_channels),
        #         ),
        #         "output": np.setdiff1d(
        #             range(original_cnn.layer2[0].out_channels),
        #             range(cnn.layer2[0].out_channels),
        #         ),
        #     }
        # )
        # indices_to_prune_conv3.append(
        #     {
        #         "input": np.setdiff1d(
        #             range(original_cnn.layer3[0].in_channels),
        #             range(cnn.layer3[0].in_channels),
        #         ),
        #         "output": np.setdiff1d(
        #             range(original_cnn.layer3[0].out_channels),
        #             range(cnn.layer3[0].out_channels),
        #         ),
        #     }
        # )
        # indices_to_prune_fc.append(
        #     {
        #         "input": np.setdiff1d(
        #             range(original_cnn.fc.in_features), range(cnn.fc.in_features)
        #         ),
        #     }
        # )

    aggregate_conv_layers(
        original_cnn.layer1[0],
        [cnn.layer1[0] for cnn in pruned_cnns],
        indices_to_prune_conv1_list,
        weights,
    )
    aggregate_conv_layers(
        original_cnn.layer2[0],
        [cnn.layer2[0] for cnn in pruned_cnns],
        indices_to_prune_conv2_list,
        weights,
    )
    aggregate_conv_layers(
        original_cnn.layer3[0],
        [cnn.layer3[0] for cnn in pruned_cnns],
        indices_to_prune_conv3_list,
        weights,
    )
    aggregate_linear_layers(
        original_cnn.fc,
        [cnn.fc for cnn in pruned_cnns],
        indices_to_prune_fc_list,
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
