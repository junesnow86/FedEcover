import copy
from typing import Dict, List

import numpy as np
import torch.nn as nn
from torchvision.models import ResNet

from modules.models import CNN, Transformer
from modules.pruning import ModelPrunedIndicesDict
from modules.utils import measure_time

from .aggregate_layers import (
    aggregate_conv_layers,
    aggregate_embedding_layers,
    aggregate_linear_layers,
    aggregate_transformer_block_layers,
    update_global_conv_layer,
    update_global_linear_layer,
)


def vanilla_federated_averaging(models, client_weights):
    model_weights = [model.state_dict() for model in models]
    assert len(model_weights) == len(client_weights), "Length mismatch"
    avg_weights = {}
    keys = model_weights[0].keys()

    for key in keys:
        layer_weights = [
            model_weight[key].clone().detach() * num
            for model_weight, num in zip(model_weights, client_weights)
        ]
        layer_weights_avg = sum(layer_weights) / sum(client_weights)
        avg_weights[key] = layer_weights_avg

    print(f"Keys aggregated: {set(keys)}")

    return avg_weights


@measure_time(repeats=1)
def aggregate_cnn(
    global_model: CNN,
    local_models: List[CNN],
    model_pruned_indices_dicts: List[ModelPrunedIndicesDict],
    client_weights: List[int],
):
    """
    Aggregate the weights of the conv and linear layers of the ResNet18 model.
    """
    aggregate_conv_layers(
        global_model.layer1[0],
        [model.layer1[0] for model in local_models],
        [pruned_indices_dict["layer1"] for pruned_indices_dict in model_pruned_indices_dicts],
        client_weights,
    )

    aggregate_conv_layers(
        global_model.layer2[0],
        [model.layer2[0] for model in local_models],
        [pruned_indices_dict["layer2"] for pruned_indices_dict in model_pruned_indices_dicts],
        client_weights,
    )

    aggregate_conv_layers(
        global_model.layer3[0],
        [model.layer3[0] for model in local_models],
        [pruned_indices_dict["layer3"] for pruned_indices_dict in model_pruned_indices_dicts],
        client_weights,
    )

    # ----- Linear layer -----
    aggregate_linear_layers(
        global_model.fc,
        [model.fc for model in local_models],
        [pruned_indices_dict["fc"] for pruned_indices_dict in model_pruned_indices_dicts],
        client_weights,
    )


@measure_time(repeats=1)
def aggregate_resnet18(
    global_model: ResNet,
    local_models: List[ResNet],
    model_pruned_indices_dicts: List[ModelPrunedIndicesDict],
    client_weights: List[int],
):
    """
    Aggregate the weights of the conv and linear layers of the ResNet18 model.
    """
    aggregate_conv_layers(
        global_model.conv1,
        [
            model.conv1[0] for model in local_models
        ],  # use index 0 because of nn.Sequential, due to DropoutScaling module
        [pruned_indices_dict["conv1"] for pruned_indices_dict in model_pruned_indices_dicts],
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
                        for pruned_indices_dict in model_pruned_indices_dicts
                    ],
                    client_weights,
                )

            downsample_key = f"{layer}.{block}.downsample.0"
            if downsample_key in model_pruned_indices_dicts[0]:
                aggregate_conv_layers(
                    global_model._modules[layer][int(block)].downsample[0],
                    [
                        model._modules[layer][int(block)].downsample[0]
                        for model in local_models
                    ],
                    [
                        pruned_indices_dict[downsample_key]
                        for pruned_indices_dict in model_pruned_indices_dicts
                    ],
                    client_weights,
                )

    # ----- Linear layer -----
    aggregate_linear_layers(
        global_model.fc,
        [model.fc for model in local_models],
        [pruned_indices_dict["fc"] for pruned_indices_dict in model_pruned_indices_dicts],
        client_weights,
    )


@measure_time(repeats=1)
def aggregate_transformer(
    global_model: Transformer,
    local_models: List[Transformer],
    model_pruned_indices_dicts: List[ModelPrunedIndicesDict],
    client_weights: List[int],
):
    num_layers = global_model.num_layers

    aggregate_embedding_layers(
        global_embedding_layer=global_model.encoder_embedding,
        pruned_embedding_layers=[model.encoder_embedding[0] for model in local_models],
        layer_pruned_indices_dicts=[
            model_pruned_indices_dict["embedding"]
            for model_pruned_indices_dict in model_pruned_indices_dicts
        ],
        client_weights=client_weights,
    )
    aggregate_embedding_layers(
        global_embedding_layer=global_model.decoder_embedding,
        pruned_embedding_layers=[model.decoder_embedding[0] for model in local_models],
        layer_pruned_indices_dicts=[
            model_pruned_indices_dict["embedding"]
            for model_pruned_indices_dict in model_pruned_indices_dicts
        ],
        client_weights=client_weights,
    )

    for block in range(num_layers):
        for type in ["encoder", "decoder"]:
            block_pruned_indices_dicts = [
                model_pruned_indices_dict[f"{type}.{block}"]
                for model_pruned_indices_dict in model_pruned_indices_dicts
            ]
            # Aggregate the block
            if type == "encoder":
                aggregate_transformer_block_layers(
                    global_block=global_model.encoder_blocks[block],
                    pruned_blocks=[
                        model.encoder_blocks[block] for model in local_models
                    ],
                    block_pruned_indices_dicts=block_pruned_indices_dicts,
                    client_weights=client_weights,
                )
            else:
                aggregate_transformer_block_layers(
                    global_block=global_model.decoder_blocks[block],
                    pruned_blocks=[
                        model.decoder_blocks[block] for model in local_models
                    ],
                    block_pruned_indices_dicts=block_pruned_indices_dicts,
                    client_weights=client_weights,
                )

    # ----- Linear layer -----
    aggregate_linear_layers(
        global_model.fc,
        [model.fc for model in local_models],
        [
            model_pruned_indices_dict["fc"]
            for model_pruned_indices_dict in model_pruned_indices_dicts
        ],
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


def recover_global_from_pruned_cnn(
    global_model: nn.Module,
    pruned_model: nn.Module,
    pruned_indices_dict: Dict[str, np.ndarray],
    scaler: float = 1.0,
):
    clone_global_model = copy.deepcopy(global_model)

    # conv1
    update_global_conv_layer(
        clone_global_model.layer1[0],
        pruned_model.layer1[0],
        pruned_indices_dict["layer1"],
        scaler,
    )

    # conv2
    update_global_conv_layer(
        clone_global_model.layer2[0],
        pruned_model.layer2[0],
        pruned_indices_dict["layer2"],
        scaler,
    )

    # conv3
    update_global_conv_layer(
        clone_global_model.layer3[0],
        pruned_model.layer3[0],
        pruned_indices_dict["layer3"],
        scaler,
    )

    # Linear layer
    update_global_linear_layer(
        clone_global_model.fc, pruned_model.fc, pruned_indices_dict["fc"], scaler
    )

    return clone_global_model


def recover_global_from_pruned_resnet18(
    global_model: nn.Module,
    pruned_model: nn.Module,
    pruned_indices_dict: Dict[str, np.ndarray],
    scaler: float = 1.0,
):
    clone_global_model = copy.deepcopy(global_model)

    # conv1
    update_global_conv_layer(
        clone_global_model.conv1,
        pruned_model.conv1[0],
        pruned_indices_dict["conv1"],
        scaler,
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
                    scaler,
                )

            downsample_key = f"{layer_name}.{block}.downsample.0"
            if downsample_key in pruned_indices_dict:
                update_global_conv_layer(
                    clone_global_model._modules[layer_name][int(block)].downsample[0],
                    pruned_model._modules[layer_name][int(block)].downsample[0],
                    pruned_indices_dict[downsample_key],
                    scaler,
                )

    # Linear layer
    update_global_linear_layer(
        clone_global_model.fc, pruned_model.fc, pruned_indices_dict["fc"], scaler
    )

    return clone_global_model


def recover_global_from_pruned_shallow_resnet(
    global_model: nn.Module,
    pruned_model: nn.Module,
    pruned_indices_dict: Dict[str, np.ndarray],
    scaler: float = 1.0,
):
    clone_global_model = copy.deepcopy(global_model)

    # conv1
    update_global_conv_layer(
        clone_global_model.conv1,
        pruned_model.conv1[0],
        pruned_indices_dict["conv1"],
        scaler,
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
                    scaler,
                )

            downsample_key = f"{layer_name}.{block}.downsample.0"
            if downsample_key in pruned_indices_dict:
                update_global_conv_layer(
                    clone_global_model._modules[layer_name][int(block)].downsample[0],
                    pruned_model._modules[layer_name][int(block)].downsample[0],
                    pruned_indices_dict[downsample_key],
                    scaler,
                )

    # Linear layer
    update_global_linear_layer(
        clone_global_model.fc, pruned_model.fc, pruned_indices_dict["fc"], scaler
    )

    return clone_global_model
