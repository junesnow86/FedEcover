from typing import Dict, List

import numpy as np
import torch


def prune_linear_layer(layer, pruned_indices=None):
    """
    Prune a linear layer by using provided pruned_indices to directly select neurons to drop.

    Parameters:
    - layer: The linear layer to prune (an instance of torch.nn.Linear).
    - pruned_indices: A dictionary with keys 'input' and 'output', indicating the indices of neurons to prune directly.

    Returns:
    - new_layer: The new linear layer with pruned neurons.
    """
    input_features = layer.in_features
    output_features = layer.out_features

    if pruned_indices is not None:
        # Make sure the pruned indices are in relative order
        input_indices_to_keep = np.setdiff1d(
            range(input_features), pruned_indices.get("input", np.array([]))
        )
        output_indices_to_keep = np.setdiff1d(
            range(output_features), pruned_indices.get("output", np.array([]))
        )

    # Extract the weights and biases for the remaining neurons
    new_weight = layer.weight.data[output_indices_to_keep, :][:, input_indices_to_keep]
    new_bias = (
        layer.bias.data[output_indices_to_keep] if layer.bias is not None else None
    )

    # Create a new Linear layer with the pruned neurons
    new_layer = torch.nn.Linear(len(input_indices_to_keep), len(output_indices_to_keep))
    new_layer.weight = torch.nn.Parameter(new_weight)
    new_layer.bias = torch.nn.Parameter(new_bias) if new_bias is not None else None

    return new_layer


def prune_conv_layer(conv_layer, pruned_indices=None):
    """
    Prune a convolution layer by using provided pruned_indices to directly select channels to drop.

    Parameters:
    - layer: The convolution layer to prune (an instance of torch.nn.Conv2d).
    - pruned_indices: A dictionary with keys 'input' and 'output', indicating the indices of channels to prune directly.

    Returns:
    - new_layer: The new convolution layer with pruned channels.
    """
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
        conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
    )
    new_conv_layer.weight = torch.nn.Parameter(new_weight)
    new_conv_layer.bias = torch.nn.Parameter(new_bias) if new_bias is not None else None

    return new_conv_layer


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
    weight_sample_accumulator = torch.zeros((global_output_size, global_input_size))
    if global_linear_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_linear_layer.bias.data)
        bias_sample_accumulator = torch.zeros(global_output_size)

    for linear_layer, pruned_indices, num_samples in zip(
        linear_layer_list, pruned_indices_list, num_samples_list
    ):
        layer_weights = linear_layer.weight.data

        pruned_input_indices = pruned_indices.get("input", np.array([]))
        pruned_output_indices = pruned_indices.get("output", np.array([]))

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
                weight_sample_accumulator[out_idx_global, in_idx_global] += num_samples

        if linear_layer.bias is not None:
            layer_bias = linear_layer.bias.data
            for out_idx_global in unpruned_output_indices:
                out_idx_layer = output_index_map[out_idx_global]
                bias_accumulator[out_idx_global] += (
                    layer_bias[out_idx_layer] * num_samples
                )
                bias_sample_accumulator[out_idx_global] += num_samples

        # Normalize the accumulated weights and biases by the number of samples
        for out_idx_global in unpruned_output_indices:
            for in_idx_global in unpruned_input_indices:
                if weight_sample_accumulator[out_idx_global, in_idx_global] > 0:
                    global_linear_layer.weight.data[out_idx_global, in_idx_global] = (
                        weight_accumulator[out_idx_global, in_idx_global]
                        / weight_sample_accumulator[out_idx_global, in_idx_global]
                    )

        if global_linear_layer.bias is not None:
            for out_idx_global in unpruned_output_indices:
                if bias_sample_accumulator[out_idx_global] > 0:
                    global_linear_layer.bias.data[out_idx_global] = (
                        bias_accumulator[out_idx_global]
                        / bias_sample_accumulator[out_idx_global]
                    )


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
    weight_sample_accumulator = torch.zeros((global_out_channels, global_in_channels))
    if global_conv_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_conv_layer.bias.data)
        bias_sample_accumulator = torch.zeros(global_out_channels)

    for conv_layer, pruned_indices, num_samples in zip(
        conv_layer_list, pruned_indices_list, num_samples_list
    ):
        layer_weights = conv_layer.weight.data

        pruned_in_indices = pruned_indices.get("input", np.array([]))
        pruned_out_indices = pruned_indices.get("output", np.array([]))

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
                weight_sample_accumulator[out_idx_global, in_idx_global] += num_samples

        if conv_layer.bias is not None:
            layer_bias = conv_layer.bias.data
            for out_idx_global in unpruned_out_indices:
                out_idx_layer = output_index_map[out_idx_global]
                bias_accumulator[out_idx_global] += (
                    layer_bias[out_idx_layer] * num_samples
                )
                bias_sample_accumulator[out_idx_global] += num_samples

        # Normalize the accumulated weights and biases by the number of samples
        for out_idx_global in unpruned_out_indices:
            for in_idx_global in unpruned_in_indices:
                if weight_sample_accumulator[out_idx_global, in_idx_global] > 0:
                    global_conv_layer.weight.data[
                        out_idx_global, in_idx_global, :, :
                    ] = (
                        weight_accumulator[out_idx_global, in_idx_global, :, :]
                        / weight_sample_accumulator[out_idx_global, in_idx_global]
                    )

        if global_conv_layer.bias is not None:
            for out_idx_global in unpruned_out_indices:
                if bias_sample_accumulator[out_idx_global].sum() > 0:
                    global_conv_layer.bias.data[out_idx_global] = (
                        bias_accumulator[out_idx_global]
                        / bias_sample_accumulator[out_idx_global]
                    )


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
