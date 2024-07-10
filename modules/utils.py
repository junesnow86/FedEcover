from typing import Dict, List

import numpy as np
import torch


# FIXME: test if single aggregation works
def prune_linear_layer(
    layer,
    p=0.5,
    prune_input=True,
    prune_output=True,
    num_neurons_to_prune=None,
    use_absolute_number=False,
):
    """
    Prune a linear layer by either randomly dropping a proportion p of input and output neurons
    or by dropping a specific number of neurons if use_absolute_number is True.

    Parameters:
    - layer: The linear layer to prune (an instance of torch.nn.Linear).
    - p: The proportion of neurons to drop.
    - num_neurons_to_prune: A dictionary with keys 'input' and 'output', indicating the number of neurons to prune.
    - use_absolute_number: A boolean indicating whether to use the absolute number of neurons to prune.

    Returns:
    - new_layer: The new linear layer with pruned neurons.
    - pruned_indices: A dictionary with keys 'input' and 'output', indicating the indices of pruned neurons.
    """
    input_features = layer.in_features
    output_features = layer.out_features
    if use_absolute_number:
        if prune_input:
            input_neurons_to_keep = input_features - num_neurons_to_prune.get(
                "input", 0
            )
        else:
            input_neurons_to_keep = input_features
        if prune_output:
            output_neurons_to_keep = output_features - num_neurons_to_prune.get(
                "output", 0
            )
        else:
            output_neurons_to_keep = output_features
    else:
        if prune_input:
            input_neurons_to_keep = int(input_features * (1 - p))
        else:
            input_neurons_to_keep = input_features
        if prune_output:
            output_neurons_to_keep = int(output_features * (1 - p))
        else:
            output_neurons_to_keep = output_features

    # Randomly select the neurons to keep
    input_indices_to_keep = np.sort(
        np.random.choice(range(input_features), input_neurons_to_keep, replace=False)
    )
    output_indices_to_keep = np.sort(
        np.random.choice(range(output_features), output_neurons_to_keep, replace=False)
    )

    # Extract the weights and biases for the remaining neurons
    new_weight = layer.weight.data[output_indices_to_keep, :][:, input_indices_to_keep]
    new_bias = (
        layer.bias.data[output_indices_to_keep] if layer.bias is not None else None
    )

    # Create a new Linear layer with the pruned neurons
    new_layer = torch.nn.Linear(input_neurons_to_keep, output_neurons_to_keep)
    new_layer.weight = torch.nn.Parameter(new_weight)
    new_layer.bias = torch.nn.Parameter(new_bias) if new_bias is not None else None

    # Record the pruned indices
    pruned_indices = {
        "input": np.setdiff1d(range(input_features), input_indices_to_keep),
        "output": np.setdiff1d(range(output_features), output_indices_to_keep),
    }

    return new_layer, pruned_indices


def prune_conv_layer(conv_layer, p=0.5, prune_input=True, prune_output=True):
    # Calculate the number of filters to keep
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels
    if prune_input:
        in_channels_to_keep = int(in_channels * (1 - p))
    else:
        in_channels_to_keep = in_channels
    if prune_output:
        out_channels_to_keep = int(out_channels * (1 - p))
    else:
        out_channels_to_keep = out_channels

    # Randomly select the filters to keep
    in_indices_to_keep = np.sort(
        np.random.choice(range(in_channels), in_channels_to_keep, replace=False)
    )
    out_indices_to_keep = np.sort(
        np.random.choice(range(out_channels), out_channels_to_keep, replace=False)
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
        in_channels_to_keep,
        out_channels_to_keep,
        conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
    )
    new_conv_layer.weight = torch.nn.Parameter(new_weight)
    new_conv_layer.bias = torch.nn.Parameter(new_bias) if new_bias is not None else None

    # Record the pruned indices
    pruned_indices = {
        "input": np.setdiff1d(range(in_channels), in_indices_to_keep),
        "output": np.setdiff1d(range(out_channels), out_indices_to_keep),
    }

    return new_conv_layer, pruned_indices


def aggregate_linear_layers(
    global_linear_layer,
    linear_layer_list,
    pruned_indices_list: List[Dict[str, np.ndarray]],
    num_samples_list,
):
    assert len(linear_layer_list) == len(
        pruned_indices_list
    ), f"Length mismatch: {len(linear_layer_list)} vs {len(pruned_indices_list)}"
    assert len(linear_layer_list) == len(
        num_samples_list
    ), f"Length mismatch: {len(linear_layer_list)} vs {len(num_samples_list)}"

    global_output_size = global_linear_layer.weight.data.shape[0]
    global_input_size = global_linear_layer.weight.data.shape[1]

    # Initialize accumulator for weights and biases with zeros
    weight_accumulator = torch.zeros_like(global_linear_layer.weight.data)
    if global_linear_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_linear_layer.bias.data)
    sample_accumulator = torch.zeros((global_output_size, global_input_size))

    for linear_layer, pruned_indices, num_samples in zip(
        linear_layer_list, pruned_indices_list, num_samples_list
    ):
        layer_weights = linear_layer.weight.data

        pruned_input_indices = pruned_indices["input"]
        pruned_output_indices = pruned_indices["output"]

        unpruned_input_indices = np.setdiff1d(
            range(global_input_size), pruned_input_indices
        )
        unpruned_output_indices = np.setdiff1d(
            range(global_output_size), pruned_output_indices
        )

        input_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_input_indices)
        }
        output_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_output_indices)
        }

        for out_idx_global in unpruned_output_indices:
            for in_idx_global in unpruned_input_indices:
                out_idx_layer = output_index_map[out_idx_global]
                in_idx_layer = input_index_map[in_idx_global]
                weight_accumulator[out_idx_global, in_idx_global] += (
                    layer_weights[out_idx_layer, in_idx_layer] * num_samples
                )
                sample_accumulator[out_idx_global, in_idx_global] += num_samples

        if linear_layer.bias is not None:
            layer_bias = linear_layer.bias.data
            for out_idx_global in unpruned_output_indices:
                out_idx_layer = output_index_map[out_idx_global]
                bias_accumulator[out_idx_global] += (
                    layer_bias[out_idx_layer] * num_samples
                )

        # Normalize the accumulated weights and biases by the number of samples
        for out_idx_global in unpruned_output_indices:
            for in_idx_global in unpruned_input_indices:
                if sample_accumulator[out_idx_global, in_idx_global] > 0:
                    global_linear_layer.weight.data[out_idx_global, in_idx_global] = (
                        weight_accumulator[out_idx_global, in_idx_global]
                        / sample_accumulator[out_idx_global, in_idx_global]
                    )
                else:
                    global_linear_layer.weight.data[out_idx_global, in_idx_global] = 0.0

        if global_linear_layer.bias is not None:
            for out_idx_global in unpruned_output_indices:
                if sample_accumulator[out_idx_global].sum() > 0:
                    global_linear_layer.bias.data[out_idx_global] = (
                        bias_accumulator[out_idx_global]
                        / sample_accumulator[out_idx_global, :].sum()
                    )
                else:
                    global_linear_layer.bias.data[out_idx_global] = 0.0


def aggregate_conv_layers(
    global_conv_layer,
    conv_layer_list,
    pruned_indices_list: List[Dict[str, np.ndarray]],
    num_samples_list,
):
    assert len(conv_layer_list) == len(
        pruned_indices_list
    ), f"Length mismatch: {len(conv_layer_list)} vs {len(pruned_indices_list)}"
    assert len(conv_layer_list) == len(
        num_samples_list
    ), f"Length mismatch: {len(conv_layer_list)} vs {len(num_samples_list)}"

    global_out_channels = global_conv_layer.out_channels
    global_in_channels = global_conv_layer.in_channels

    # Initialize accumulator for weights and biases with zeros
    weight_accumulator = torch.zeros_like(global_conv_layer.weight.data)
    if global_conv_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_conv_layer.bias.data)
    sample_accumulator = torch.zeros((global_out_channels, global_in_channels))

    for conv_layer, pruned_indices, num_samples in zip(
        conv_layer_list, pruned_indices_list, num_samples_list
    ):
        layer_weights = conv_layer.weight.data

        pruned_in_indices = pruned_indices["input"]
        pruned_out_indices = pruned_indices["output"]

        unpruned_in_indices = np.setdiff1d(range(global_in_channels), pruned_in_indices)
        unpruned_out_indices = np.setdiff1d(
            range(global_out_channels), pruned_out_indices
        )

        input_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_in_indices)
        }
        output_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_out_indices)
        }

        for out_idx_global in unpruned_out_indices:
            for in_idx_global in unpruned_in_indices:
                out_idx_layer = output_index_map[out_idx_global]
                in_idx_layer = input_index_map[in_idx_global]
                weight_accumulator[out_idx_global, in_idx_global, :, :] += (
                    layer_weights[out_idx_layer, in_idx_layer, :, :] * num_samples
                )
                sample_accumulator[out_idx_global, in_idx_global] += num_samples

        if conv_layer.bias is not None:
            layer_bias = conv_layer.bias.data
            for out_idx_global in unpruned_out_indices:
                out_idx_layer = output_index_map[out_idx_global]
                bias_accumulator[out_idx_global] += (
                    layer_bias[out_idx_layer] * num_samples
                )

        # Normalize the accumulated weights and biases by the number of samples
        for out_idx_global in unpruned_out_indices:
            for in_idx_global in unpruned_in_indices:
                if sample_accumulator[out_idx_global, in_idx_global] > 0:
                    global_conv_layer.weight.data[
                        out_idx_global, in_idx_global, :, :
                    ] = (
                        weight_accumulator[out_idx_global, in_idx_global, :, :]
                        / sample_accumulator[out_idx_global, in_idx_global]
                    )
                else:
                    global_conv_layer.weight.data[
                        out_idx_global, in_idx_global, :, :
                    ] = 0.0

        if global_conv_layer.bias is not None:
            for out_idx_global in unpruned_out_indices:
                if sample_accumulator[out_idx_global].sum() > 0:
                    global_conv_layer.bias.data[out_idx_global] = (
                        bias_accumulator[out_idx_global]
                        / sample_accumulator[out_idx_global, :].sum()
                    )
                else:
                    global_conv_layer.bias.data[out_idx_global] = 0.0


def federated_averaging(model_weights, sample_numbers):
    assert len(model_weights) == len(sample_numbers), "Length mismatch"
    avg_weights = {}
    keys = model_weights[0].keys()

    for key in keys:
        layer_weights = [
            model_weight[key].clone().detach() * num
            for model_weight, num in zip(model_weights, sample_numbers)
        ]
        layer_weights_avg = sum(layer_weights) / sum(sample_numbers)
        avg_weights[key] = layer_weights_avg

    return avg_weights


def hetero_federated_averaging():
    pass


def calculate_model_size(model):
    total_params = 0
    for param in model.parameters():
        # Multiply the size of each dimension to get the total number of elements
        total_params += param.numel()

    # Assuming each parameter is a 32-bit float
    memory_bytes = total_params * 4
    memory_kilobytes = memory_bytes / 1024
    memory_megabytes = memory_kilobytes / 1024

    print(f"Total parameters: {total_params}")
    print(
        f"Memory Usage: {memory_bytes} bytes ({memory_kilobytes:.2f} KB / {memory_megabytes:.2f} MB)"
    )
