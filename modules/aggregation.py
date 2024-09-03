import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet

from modules.models import CNN
from modules.utils import measure_time


def vanilla_federated_averaging(models, sample_numbers):
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

        # for out_idx_layer, out_idx_global in enumerate(unpruned_output_indices):
        #     for in_idx_layer, in_idx_global in enumerate(unpruned_input_indices):
        #         weight_accumulator[out_idx_global, in_idx_global] += (
        #             layer_weight[out_idx_layer, in_idx_layer] * num_samples
        #         )
        #         sample_accumulator_weight[out_idx_global, in_idx_global] += num_samples
        weight_accumulator[np.ix_(unpruned_output_indices, unpruned_input_indices)] += (
            layer_weight[
                np.ix_(
                    range(len(unpruned_output_indices)),
                    range(len(unpruned_input_indices)),
                )
            ]
            * num_samples
        )
        sample_accumulator_weight[
            np.ix_(unpruned_output_indices, unpruned_input_indices)
        ] += num_samples

        if linear_layer.bias is not None:
            layer_bias = linear_layer.bias.data
            # for out_idx_layer, out_idx_global in enumerate(unpruned_output_indices):
            #     bias_accumulator[out_idx_global] += (
            #         layer_bias[out_idx_layer] * num_samples
            #     )
            #     sample_accumulator_bias[out_idx_global] += num_samples
            bias_accumulator[unpruned_output_indices] += (
                layer_bias[range(len(unpruned_output_indices))] * num_samples
            )
            sample_accumulator_bias[unpruned_output_indices] += num_samples

    # Normalize the accumulated weights and biases by the number of samples after processing all layers
    # for out_idx_global in range(global_output_size):
    #     for in_idx_global in range(global_input_size):
    #         if sample_accumulator_weight[out_idx_global, in_idx_global] > 0:
    #             global_linear_layer.weight.data[out_idx_global, in_idx_global] = (
    #                 weight_accumulator[out_idx_global, in_idx_global]
    #                 / sample_accumulator_weight[out_idx_global, in_idx_global]
    #             )
    nonzero_indices = sample_accumulator_weight > 0
    global_linear_layer.weight.data[nonzero_indices] = (
        weight_accumulator[nonzero_indices] / sample_accumulator_weight[nonzero_indices]
    )

    if global_linear_layer.bias is not None:
        # for out_idx_global in range(global_output_size):
        #     if sample_accumulator_bias[out_idx_global] > 0:
        #         global_linear_layer.bias.data[out_idx_global] = (
        #             bias_accumulator[out_idx_global]
        #             / sample_accumulator_bias[out_idx_global]
        #         )
        nonzero_indices = sample_accumulator_bias > 0
        global_linear_layer.bias.data[nonzero_indices] = (
            bias_accumulator[nonzero_indices] / sample_accumulator_bias[nonzero_indices]
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

        # for out_idx_layer, out_idx_global in enumerate(unpruned_out_indices):
        #     for in_idx_layer, in_idx_global in enumerate(unpruned_in_indices):
        #         weight_accumulator[out_idx_global, in_idx_global, :, :] += (
        #             layer_weight[out_idx_layer, in_idx_layer, :, :] * num_samples
        #         )
        #         sample_accumulator_weight[out_idx_global, in_idx_global] += num_samples
        weight_accumulator[np.ix_(unpruned_out_indices, unpruned_in_indices)] += (
            layer_weight[
                np.ix_(
                    range(len(unpruned_out_indices)),
                    range(len(unpruned_in_indices)),
                )
            ]
            * num_samples
        )
        sample_accumulator_weight[
            np.ix_(unpruned_out_indices, unpruned_in_indices)
        ] += num_samples

        if conv_layer.bias is not None:
            layer_bias = conv_layer.bias.data
            # for out_idx_layer, out_idx_global in enumerate(unpruned_out_indices):
            #     bias_accumulator[out_idx_global] += (
            #         layer_bias[out_idx_layer] * num_samples
            #     )
            #     sample_accumulator_bias[out_idx_global] += num_samples
            bias_accumulator[unpruned_out_indices] += (
                layer_bias[range(len(unpruned_out_indices))] * num_samples
            )
            sample_accumulator_bias[unpruned_out_indices] += num_samples

    # Normalize the accumulated weights and biases by the number of samples after processing all layers
    for out_idx_global in range(global_out_channels):
        for in_idx_global in range(global_in_channels):
            if sample_accumulator_weight[out_idx_global, in_idx_global] > 0:
                global_conv_layer.weight.data[out_idx_global, in_idx_global, :, :] = (
                    weight_accumulator[out_idx_global, in_idx_global, :, :]
                    / sample_accumulator_weight[out_idx_global, in_idx_global]
                )
    nonzero_indices = sample_accumulator_weight > 0
    global_conv_layer.weight.data[nonzero_indices] = (
        weight_accumulator[nonzero_indices]
        / sample_accumulator_weight[nonzero_indices][:, None, None]
    )

    if global_conv_layer.bias is not None:
        # for out_idx_global in range(global_out_channels):
        #     if sample_accumulator_bias[out_idx_global] > 0:
        #         global_conv_layer.bias.data[out_idx_global] = (
        #             bias_accumulator[out_idx_global]
        #             / sample_accumulator_bias[out_idx_global]
        #         )
        nonzero_indices = sample_accumulator_bias > 0
        global_conv_layer.bias.data[nonzero_indices] = (
            bias_accumulator[nonzero_indices] / sample_accumulator_bias[nonzero_indices]
        )


def aggregate_cnn(
    original_cnn: CNN,
    pruned_cnn_list: List[CNN],
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


@measure_time(repeats=1)
def aggregate_resnet18(
    global_model: ResNet,
    local_models: List[ResNet],
    client_weights: List[int],
    pruned_indices_dicts: List[Dict[str, Dict[str, np.ndarray]]],
):
    """
    Aggregate the weights of the conv and linear layers of the ResNet18 model.
    """
    print("Using `aggregate_resnet18`")

    aggregate_conv_layers(
        global_model.conv1,
        [model.conv1[0] for model in local_models],
        [pruned_indices_dict["conv1"] for pruned_indices_dict in pruned_indices_dicts],
        client_weights,
    )

    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        for block in ["0", "1"]:
            for conv in ["conv1", "conv2"]:
                key = f"{layer}.{block}.{conv}"
                aggregate_conv_layers(
                    global_model._modules[layer][int(block)]._modules[conv],
                    [
                        model._modules[layer][int(block)]._modules[conv][0]
                        for model in local_models
                    ],
                    [
                        pruned_indices_dict[key]
                        for pruned_indices_dict in pruned_indices_dicts
                    ],
                    client_weights,
                )

            downsample_key = f"{layer}.{block}.downsample.0"
            if downsample_key in pruned_indices_dicts[0]:
                aggregate_conv_layers(
                    global_model._modules[layer][int(block)].downsample[0],
                    [
                        model._modules[layer][int(block)].downsample[0]
                        for model in local_models
                    ],
                    [
                        pruned_indices_dict[downsample_key]
                        for pruned_indices_dict in pruned_indices_dicts
                    ],
                    client_weights,
                )

    # ----- Linear layer -----
    aggregate_linear_layers(
        global_model.fc,
        [model.fc for model in local_models],
        [pruned_indices_dict["fc"] for pruned_indices_dict in pruned_indices_dicts],
        client_weights,
    )


@measure_time(repeats=1)
def aggregate_resnet18_vanilla(
    global_model: ResNet,
    local_models: List[ResNet],
    client_weights: List[int],
    pruned_indices_dicts: List[Dict[str, Dict[str, np.ndarray]]],
):
    """
    Aggregate the weights of the conv and linear layers of the ResNet18 model.
    """
    print("Using `aggregate_resnet18`")

    aggregate_conv_layers(
        global_model.conv1[0],
        [model.conv1[0] for model in local_models],
        [pruned_indices_dict["conv1"] for pruned_indices_dict in pruned_indices_dicts],
        client_weights,
    )

    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        for block in ["0", "1"]:
            for conv in ["conv1", "conv2"]:
                key = f"{layer}.{block}.{conv}"
                aggregate_conv_layers(
                    global_model._modules[layer][int(block)]._modules[conv][0],
                    [
                        model._modules[layer][int(block)]._modules[conv][0]
                        for model in local_models
                    ],
                    [
                        pruned_indices_dict[key]
                        for pruned_indices_dict in pruned_indices_dicts
                    ],
                    client_weights,
                )

            downsample_key = f"{layer}.{block}.downsample.0"
            if downsample_key in pruned_indices_dicts[0]:
                aggregate_conv_layers(
                    global_model._modules[layer][int(block)].downsample[0],
                    [
                        model._modules[layer][int(block)].downsample[0]
                        for model in local_models
                    ],
                    [
                        pruned_indices_dict[downsample_key]
                        for pruned_indices_dict in pruned_indices_dicts
                    ],
                    client_weights,
                )

    # ----- Linear layer -----
    aggregate_linear_layers(
        global_model.fc,
        [model.fc for model in local_models],
        [pruned_indices_dict["fc"] for pruned_indices_dict in pruned_indices_dicts],
        client_weights,
    )


@measure_time(repeats=1)
def aggregate_shallow_resnet(
    global_model: ResNet,
    local_models: List[ResNet],
    client_weights: List[int],
    pruned_indices_dicts: List[Dict[str, Dict[str, np.ndarray]]],
):
    """
    Aggregate the weights of the conv and linear layers of the ResNet18 model.
    """
    print("Using `aggregate_resnet18`")

    aggregate_conv_layers(
        global_model.conv1,
        [model.conv1[0] for model in local_models],
        [pruned_indices_dict["conv1"] for pruned_indices_dict in pruned_indices_dicts],
        client_weights,
    )

    for layer in ["layer1", "layer2"]:
    # for layer in ["layer1"]:
        # for block in ["0", "1"]:
        for block in ["0"]:
            for conv in ["conv1", "conv2"]:
                key = f"{layer}.{block}.{conv}"
                aggregate_conv_layers(
                    global_model._modules[layer][int(block)]._modules[conv],
                    [
                        model._modules[layer][int(block)]._modules[conv][0]
                        for model in local_models
                    ],
                    [
                        pruned_indices_dict[key]
                        for pruned_indices_dict in pruned_indices_dicts
                    ],
                    client_weights,
                )

            downsample_key = f"{layer}.{block}.downsample.0"
            if downsample_key in pruned_indices_dicts[0]:
                aggregate_conv_layers(
                    global_model._modules[layer][int(block)].downsample[0],
                    [
                        model._modules[layer][int(block)].downsample[0]
                        for model in local_models
                    ],
                    [
                        pruned_indices_dict[downsample_key]
                        for pruned_indices_dict in pruned_indices_dicts
                    ],
                    client_weights,
                )

    # ----- Linear layer -----
    aggregate_linear_layers(
        global_model.fc,
        [model.fc for model in local_models],
        [pruned_indices_dict["fc"] for pruned_indices_dict in pruned_indices_dicts],
        client_weights,
    )


def update_global_linear_layer(global_layer, local_layer, pruned_indices_dict):
    """Update the global layer inplacely with the weights and biases of the local layer.

    Args:
        global_layer: The global layer to be updated.
        local_layer: The local layer which is pruned from the global layer, and then updated.
        pruned_indices_dict: A dictionary containing the indices indicating which neurons have been pruned from global layer.

    """
    # Update the weights
    new_weight = global_layer.weight.detach().clone()
    new_weight[
        np.ix_(
            np.sort(
                np.setdiff1d(
                    range(global_layer.out_features),
                    pruned_indices_dict.get("output", np.array([])),
                )
            ),
            np.sort(
                np.setdiff1d(
                    range(global_layer.in_features),
                    pruned_indices_dict.get("input", np.array([])),
                )
            ),
        )
    ] = local_layer.weight.detach().clone()

    # Update the biases
    if global_layer.bias is not None:
        new_bias = global_layer.bias.detach().clone()
        new_bias[
            np.sort(
                np.setdiff1d(
                    range(global_layer.out_features),
                    pruned_indices_dict.get("output", np.array([])),
                )
            )
        ] = local_layer.bias.detach().clone()
    else:
        new_bias = None

    # Copy the updated weights and biases back to the global layer
    global_layer.weight.data.copy_(new_weight)
    if new_bias is not None:
        global_layer.bias.data.copy_(new_bias)


def update_global_conv_layer(global_layer, local_layer, pruned_indices_dict):
    """Update the global layer inplacely with the weights and biases of the local layer.

    Args:
        global_layer: The global layer to be updated.
        local_layer: The local layer which is pruned from the global layer, and then updated.
        pruned_indices_dict: A dictionary containing the indices indicating which neurons have been pruned from global layer.
    """
    # Update the weights
    new_weight = global_layer.weight.detach().clone()
    new_weight[
        np.ix_(
            np.sort(
                np.setdiff1d(
                    range(global_layer.out_channels),
                    pruned_indices_dict.get("output", np.array([])),
                )
            ),
            np.sort(
                np.setdiff1d(
                    range(global_layer.in_channels),
                    pruned_indices_dict.get("input", np.array([])),
                )
            ),
        )
    ] = local_layer.weight.detach().clone()

    # Update the biases
    if global_layer.bias is not None:
        new_bias = global_layer.bias.detach().clone()
        new_bias[
            np.sort(
                np.setdiff1d(
                    range(global_layer.out_channels),
                    pruned_indices_dict.get("output", np.array([])),
                )
            )
        ] = local_layer.bias.detach().clone()
    else:
        new_bias = None

    # Copy the updated weights and biases back to the global layer
    global_layer.weight.data.copy_(new_weight)
    if new_bias is not None:
        global_layer.bias.data.copy_(new_bias)


def recover_global_from_pruned_cnn(
    global_model: nn.Module,
    pruned_model: nn.Module,
    pruned_indices_dict: Dict[str, np.ndarray],
):
    clone_global_model = copy.deepcopy(global_model)

    # conv1
    update_global_conv_layer(
        clone_global_model.layer1[0],
        pruned_model.layer1[0],
        pruned_indices_dict["layer1"],
    )

    # conv2
    update_global_conv_layer(
        clone_global_model.layer2[0],
        pruned_model.layer2[0],
        pruned_indices_dict["layer2"],
    )

    # conv3
    update_global_conv_layer(
        clone_global_model.layer3[0],
        pruned_model.layer3[0],
        pruned_indices_dict["layer3"],
    )

    # Linear layer
    update_global_linear_layer(
        clone_global_model.fc, pruned_model.fc, pruned_indices_dict["fc"]
    )

    return clone_global_model


def recover_global_from_pruned_resnet18(
    global_model: nn.Module,
    pruned_model: nn.Module,
    pruned_indices_dict: Dict[str, np.ndarray],
):
    clone_global_model = copy.deepcopy(global_model)

    # conv1
    update_global_conv_layer(
        clone_global_model.conv1, pruned_model.conv1[0], pruned_indices_dict["conv1"]
    )

    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    blocks = ["0", "1"]
    convs = ["conv1", "conv2"]

    for layer_name in layer_names:
        for block in blocks:
            for conv in convs:
                key = f"{layer_name}.{block}.{conv}"
                update_global_conv_layer(
                    clone_global_model._modules[layer_name][int(block)]._modules[conv],
                    pruned_model._modules[layer_name][int(block)]._modules[conv][0],
                    pruned_indices_dict[key],
                )

            downsample_key = f"{layer_name}.{block}.downsample.0"
            if downsample_key in pruned_indices_dict:
                update_global_conv_layer(
                    clone_global_model._modules[layer_name][int(block)].downsample[0],
                    pruned_model._modules[layer_name][int(block)].downsample[0],
                    pruned_indices_dict[downsample_key],
                )

    # Linear layer
    update_global_linear_layer(
        clone_global_model.fc, pruned_model.fc, pruned_indices_dict["fc"]
    )

    return clone_global_model


def recover_global_from_pruned_shallow_resnet(
    global_model: nn.Module,
    pruned_model: nn.Module,
    pruned_indices_dict: Dict[str, np.ndarray],
):
    clone_global_model = copy.deepcopy(global_model)

    # conv1
    update_global_conv_layer(
        clone_global_model.conv1, pruned_model.conv1[0], pruned_indices_dict["conv1"]
    )

    # layer_names = ["layer1", "layer2", "layer3", "layer4"]
    layer_names = ["layer1", "layer2"]
    blocks = ["0"]
    convs = ["conv1", "conv2"]

    for layer_name in layer_names:
        for block in blocks:
            for conv in convs:
                key = f"{layer_name}.{block}.{conv}"
                update_global_conv_layer(
                    clone_global_model._modules[layer_name][int(block)]._modules[conv],
                    pruned_model._modules[layer_name][int(block)]._modules[conv][0],
                    pruned_indices_dict[key],
                )

            downsample_key = f"{layer_name}.{block}.downsample.0"
            if downsample_key in pruned_indices_dict:
                update_global_conv_layer(
                    clone_global_model._modules[layer_name][int(block)].downsample[0],
                    pruned_model._modules[layer_name][int(block)].downsample[0],
                    pruned_indices_dict[downsample_key],
                )

    # Linear layer
    update_global_linear_layer(
        clone_global_model.fc, pruned_model.fc, pruned_indices_dict["fc"]
    )

    return clone_global_model
