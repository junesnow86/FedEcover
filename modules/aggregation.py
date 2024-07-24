from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from modules.models import CNN


def aggregate_linear_layers(
    global_linear_layer: nn.Linear,
    linear_layer_list: List[nn.Linear],
    pruned_indices_list: List[Dict[str, np.ndarray]],
    num_samples_list: List[int],
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
    sample_accumulator_weight = torch.zeros((global_output_size, global_input_size))
    if global_linear_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_linear_layer.bias.data)
        sample_accumulator_bias = torch.zeros(global_output_size)

    for linear_layer, pruned_indices, num_samples in zip(
        linear_layer_list, pruned_indices_list, num_samples_list
    ):
        layer_weight = linear_layer.weight.data

        unpruned_input_indices = np.setdiff1d(
            range(global_input_size), pruned_indices.get("input", np.array([]))
        )
        unpruned_output_indices = np.setdiff1d(
            range(global_output_size), pruned_indices.get("output", np.array([]))
        )

        for out_idx_layer, out_idx_global in enumerate(unpruned_output_indices):
            for in_idx_layer, in_idx_global in enumerate(unpruned_input_indices):
                weight_accumulator[out_idx_global, in_idx_global] += (
                    layer_weight[out_idx_layer, in_idx_layer] * num_samples
                )
                sample_accumulator_weight[out_idx_global, in_idx_global] += num_samples

        if linear_layer.bias is not None:
            layer_bias = linear_layer.bias.data
            for out_idx_layer, out_idx_global in enumerate(unpruned_output_indices):
                bias_accumulator[out_idx_global] += (
                    layer_bias[out_idx_layer] * num_samples
                )
                sample_accumulator_bias[out_idx_global] += num_samples

    # Normalize the accumulated weights and biases by the number of samples after processing all layers
    for out_idx_global in range(global_output_size):
        for in_idx_global in range(global_input_size):
            if sample_accumulator_weight[out_idx_global, in_idx_global] > 0:
                global_linear_layer.weight.data[out_idx_global, in_idx_global] = (
                    weight_accumulator[out_idx_global, in_idx_global]
                    / sample_accumulator_weight[out_idx_global, in_idx_global]
                )

    if global_linear_layer.bias is not None:
        for out_idx_global in range(global_output_size):
            if sample_accumulator_bias[out_idx_global] > 0:
                global_linear_layer.bias.data[out_idx_global] = (
                    bias_accumulator[out_idx_global]
                    / sample_accumulator_bias[out_idx_global]
                )


def aggregate_conv_layers(
    global_conv_layer: nn.Conv2d,
    conv_layer_list: List[nn.Conv2d],
    pruned_indices_list: List[Dict[str, np.ndarray]],
    num_samples_list: List[int],
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
    sample_accumulator_weight = torch.zeros((global_out_channels, global_in_channels))
    if global_conv_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_conv_layer.bias.data)
        sample_accumulator_bias = torch.zeros(global_out_channels)

    for conv_layer, pruned_indices, num_samples in zip(
        conv_layer_list, pruned_indices_list, num_samples_list
    ):
        layer_weight = conv_layer.weight.data

        unpruned_in_indices = np.setdiff1d(
            range(global_in_channels), pruned_indices.get("input", np.array([]))
        )
        unpruned_out_indices = np.setdiff1d(
            range(global_out_channels), pruned_indices.get("output", np.array([]))
        )

        for out_idx_layer, out_idx_global in enumerate(unpruned_out_indices):
            for in_idx_layer, in_idx_global in enumerate(unpruned_in_indices):
                weight_accumulator[out_idx_global, in_idx_global, :, :] += (
                    layer_weight[out_idx_layer, in_idx_layer, :, :] * num_samples
                )
                sample_accumulator_weight[out_idx_global, in_idx_global] += num_samples

        if conv_layer.bias is not None:
            layer_bias = conv_layer.bias.data
            for out_idx_layer, out_idx_global in enumerate(unpruned_out_indices):
                bias_accumulator[out_idx_global] += (
                    layer_bias[out_idx_layer] * num_samples
                )
                sample_accumulator_bias[out_idx_global] += num_samples

    # Normalize the accumulated weights and biases by the number of samples after processing all layers
    for out_idx_global in range(global_out_channels):
        for in_idx_global in range(global_in_channels):
            if sample_accumulator_weight[out_idx_global, in_idx_global] > 0:
                global_conv_layer.weight.data[out_idx_global, in_idx_global, :, :] = (
                    weight_accumulator[out_idx_global, in_idx_global, :, :]
                    / sample_accumulator_weight[out_idx_global, in_idx_global]
                )

    if global_conv_layer.bias is not None:
        for out_idx_global in range(global_out_channels):
            if sample_accumulator_bias[out_idx_global] > 0:
                global_conv_layer.bias.data[out_idx_global] = (
                    bias_accumulator[out_idx_global]
                    / sample_accumulator_bias[out_idx_global]
                )


def aggregate_cnn(
    original_cnn: CNN,
    pruned_cnn_list: List[CNN],
    # dropout_rate_list: List[float],
    num_samples_list: List[int],
    **indices_to_prune,
):
    indices_to_prune_conv1 = indices_to_prune.get("indices_to_prune_conv1", [])
    indices_to_prune_conv2 = indices_to_prune.get("indices_to_prune_conv2", [])
    indices_to_prune_conv3 = indices_to_prune.get("indices_to_prune_conv3", [])
    indices_to_prune_fc = indices_to_prune.get("indices_to_prune_fc", [])

    aggregate_conv_layers(
        original_cnn.layer1[0],
        [pruned_cnn.layer1[0] for pruned_cnn in pruned_cnn_list],
        indices_to_prune_conv1,
        num_samples_list,
    )
    aggregate_conv_layers(
        original_cnn.layer2[0],
        [pruned_cnn.layer2[0] for pruned_cnn in pruned_cnn_list],
        indices_to_prune_conv2,
        num_samples_list,
    )
    aggregate_conv_layers(
        original_cnn.layer3[0],
        [pruned_cnn.layer3[0] for pruned_cnn in pruned_cnn_list],
        indices_to_prune_conv3,
        num_samples_list,
    )
    aggregate_linear_layers(
        original_cnn.fc,
        [pruned_cnn.fc for pruned_cnn in pruned_cnn_list],
        indices_to_prune_fc,
        num_samples_list,
    )


def vanilla_federated_averaging(global_model, models, sample_numbers):
    model_weights = [model.state_dict() for model in models]
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

    print(f"Keys aggregated: {set(keys)}")

    return avg_weights
