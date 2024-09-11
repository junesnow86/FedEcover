from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

from modules.models import CNN
from modules.models.transformer import (
    DecoderBlock,
    EncoderBlock,
    FeedForward,
    MultiHeadAttention,
)
from modules.pruning import (
    BlockPrunedIndicesDict,
    LayerPrunedIndicesDict,
)


def aggregate_linear_layers(
    global_linear_layer: nn.Linear,
    pruned_linear_layers: List[nn.Linear],
    layer_pruned_indices_dicts: List[LayerPrunedIndicesDict],
    client_weights: List[int],
):
    assert (
        len(pruned_linear_layers) == len(layer_pruned_indices_dicts)
    ), f"Length mismatch: {len(pruned_linear_layers)} vs {len(layer_pruned_indices_dicts)}"
    assert len(pruned_linear_layers) == len(
        client_weights
    ), f"Length mismatch: {len(pruned_linear_layers)} vs {len(client_weights)}"

    global_output_size = global_linear_layer.weight.data.shape[0]
    global_input_size = global_linear_layer.weight.data.shape[1]

    # Initialize accumulator for weights and biases with zeros
    weight_accumulator = torch.zeros_like(global_linear_layer.weight.data)
    sample_accumulator_weight = torch.zeros((global_output_size, global_input_size))
    if global_linear_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_linear_layer.bias.data)
        sample_accumulator_bias = torch.zeros(global_output_size)

    for linear_layer, pruned_indices, num_samples in zip(
        pruned_linear_layers, layer_pruned_indices_dicts, client_weights
    ):
        layer_weight = linear_layer.weight.data

        unpruned_input_indices = np.sort(
            np.setdiff1d(
                range(global_input_size), pruned_indices.get("input", np.array([]))
            )
        )
        unpruned_output_indices = np.sort(
            np.setdiff1d(
                range(global_output_size), pruned_indices.get("output", np.array([]))
            )
        )

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
            bias_accumulator[unpruned_output_indices] += (
                layer_bias[range(len(unpruned_output_indices))] * num_samples
            )
            sample_accumulator_bias[unpruned_output_indices] += num_samples

    nonzero_indices = sample_accumulator_weight > 0
    global_linear_layer.weight.data[nonzero_indices] = (
        weight_accumulator[nonzero_indices] / sample_accumulator_weight[nonzero_indices]
    )

    if global_linear_layer.bias is not None:
        nonzero_indices = sample_accumulator_bias > 0
        global_linear_layer.bias.data[nonzero_indices] = (
            bias_accumulator[nonzero_indices] / sample_accumulator_bias[nonzero_indices]
        )


def aggregate_conv_layers(
    global_conv_layer: nn.Conv2d,
    pruned_conv_layers: List[nn.Conv2d],
    layer_pruned_indices_dicts: List[LayerPrunedIndicesDict],
    client_weights: List[int],
):
    assert (
        len(pruned_conv_layers) == len(layer_pruned_indices_dicts)
    ), f"Length mismatch: {len(pruned_conv_layers)} vs {len(layer_pruned_indices_dicts)}"
    assert len(pruned_conv_layers) == len(
        client_weights
    ), f"Length mismatch: {len(pruned_conv_layers)} vs {len(client_weights)}"

    global_out_channels = global_conv_layer.out_channels
    global_in_channels = global_conv_layer.in_channels

    # Initialize accumulator for weights and biases with zeros
    weight_accumulator = torch.zeros_like(global_conv_layer.weight.data)
    sample_accumulator_weight = torch.zeros((global_out_channels, global_in_channels))
    if global_conv_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_conv_layer.bias.data)
        sample_accumulator_bias = torch.zeros(global_out_channels)

    for conv_layer, pruned_indices, num_samples in zip(
        pruned_conv_layers, layer_pruned_indices_dicts, client_weights
    ):
        layer_weight = conv_layer.weight.data

        unpruned_in_indices = np.sort(
            np.setdiff1d(
                range(global_in_channels), pruned_indices.get("input", np.array([]))
            )
        )
        unpruned_out_indices = np.sort(
            np.setdiff1d(
                range(global_out_channels), pruned_indices.get("output", np.array([]))
            )
        )

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
        nonzero_indices = sample_accumulator_bias > 0
        global_conv_layer.bias.data[nonzero_indices] = (
            bias_accumulator[nonzero_indices] / sample_accumulator_bias[nonzero_indices]
        )


def aggregate_embedding_layers(
    global_embedding_layer: nn.Embedding,
    pruned_embedding_layers: List[nn.Embedding],
    layer_pruned_indices_dicts: List[LayerPrunedIndicesDict],
    client_weights: List[int],
):
    num_clients = len(pruned_embedding_layers)
    assert len(layer_pruned_indices_dicts) == num_clients
    assert len(client_weights) == num_clients

    global_num_embeddings = global_embedding_layer.num_embeddings
    global_embedding_dim = global_embedding_layer.embedding_dim

    weight_accumulator = torch.zeros_like(global_embedding_layer.weight.data)
    client_weight_accumulator = torch.zeros(global_num_embeddings, global_embedding_dim)

    for pruned_embedding_layer, pruned_indices_dict, client_weight in zip(
        pruned_embedding_layers, layer_pruned_indices_dicts, client_weights
    ):
        pruned_layer_weight = pruned_embedding_layer.weight.data
        in_indices_keep = np.array(range(global_num_embeddings))
        out_indices_keep = np.sort(
            np.setdiff1d(
                range(global_embedding_dim),
                pruned_indices_dict.get("output", np.array([])),
            )
        )
        weight_accumulator[np.ix_(in_indices_keep, out_indices_keep)] += (
            pruned_layer_weight * client_weight
        )
        client_weight_accumulator[np.ix_(in_indices_keep, out_indices_keep)] += (
            client_weight
        )

    nonzero_indices = client_weight_accumulator > 0
    global_embedding_layer.weight.data[nonzero_indices] = (
        weight_accumulator[nonzero_indices] / client_weight_accumulator[nonzero_indices]
    )


def aggregate_multihead_attention_layers(
    global_mha_layer: MultiHeadAttention,
    pruned_mha_layers: List[MultiHeadAttention],
    layer_pruned_indices_dicts: List[LayerPrunedIndicesDict],
    client_weights: List[int],
):
    num_clients = len(pruned_mha_layers)
    assert len(layer_pruned_indices_dicts) == num_clients
    assert len(client_weights) == num_clients

    aggregate_linear_layers(
        global_mha_layer.W_q,
        [pruned_mha_layer.W_q[0] for pruned_mha_layer in pruned_mha_layers],
        layer_pruned_indices_dicts,
        client_weights,
    )
    aggregate_linear_layers(
        global_mha_layer.W_k,
        [pruned_mha_layer.W_k[0] for pruned_mha_layer in pruned_mha_layers],
        layer_pruned_indices_dicts,
        client_weights,
    )
    aggregate_linear_layers(
        global_mha_layer.W_v,
        [pruned_mha_layer.W_v[0] for pruned_mha_layer in pruned_mha_layers],
        layer_pruned_indices_dicts,
        client_weights,
    )
    aggregate_linear_layers(
        global_mha_layer.W_o,
        [pruned_mha_layer.W_o[0] for pruned_mha_layer in pruned_mha_layers],
        [
            {
                "input": layer_pruned_indices_dict.get("output", np.array([])),
                "output": layer_pruned_indices_dict.get("input", np.array([])),
            }
            for layer_pruned_indices_dict in layer_pruned_indices_dicts
        ],
        client_weights,
    )


def aggregate_feedforward_layers(
    global_ff_layer: FeedForward,
    pruned_ff_layers: List[FeedForward],
    layer_pruned_indices_dicts: List[LayerPrunedIndicesDict],
    client_weights: List[int],
):
    num_clients = len(pruned_ff_layers)
    assert len(layer_pruned_indices_dicts) == num_clients
    assert len(client_weights) == num_clients

    aggregate_linear_layers(
        global_ff_layer.fc1,
        [pruned_ff_layer.fc1[0] for pruned_ff_layer in pruned_ff_layers],
        layer_pruned_indices_dicts,
        client_weights,
    )
    aggregate_linear_layers(
        global_ff_layer.fc2,
        [pruned_ff_layer.fc2[0] for pruned_ff_layer in pruned_ff_layers],
        [
            {
                "input": layer_pruned_indices_dict.get("output", np.array([])),
                "output": layer_pruned_indices_dict.get("input", np.array([])),
            }
            for layer_pruned_indices_dict in layer_pruned_indices_dicts
        ],
        client_weights,
    )


def aggregate_transformer_block_layers(
    global_block: Union[EncoderBlock, DecoderBlock],
    pruned_blocks: Union[List[EncoderBlock], List[DecoderBlock]],
    block_pruned_indices_dicts: List[BlockPrunedIndicesDict],
    client_weights: List[int],
):
    num_clients = len(pruned_blocks)
    assert len(block_pruned_indices_dicts) == num_clients
    assert len(client_weights) == num_clients

    aggregate_multihead_attention_layers(
        global_block.self_attn,
        [pruned_block.self_attn for pruned_block in pruned_blocks],
        [
            block_pruned_indices_dict["self_attn"]
            for block_pruned_indices_dict in block_pruned_indices_dicts
        ],
        client_weights,
    )

    if isinstance(global_block, DecoderBlock):
        aggregate_multihead_attention_layers(
            global_block.cross_attn,
            [pruned_block.cross_attn for pruned_block in pruned_blocks],
            [
                block_pruned_indices_dict["cross_attn"]
                for block_pruned_indices_dict in block_pruned_indices_dicts
            ],
            client_weights,
        )

    aggregate_feedforward_layers(
        global_block.ffn,
        [pruned_block.ffn for pruned_block in pruned_blocks],
        [
            block_pruned_indices_dict["feedforward"]
            for block_pruned_indices_dict in block_pruned_indices_dicts
        ],
        client_weights,
    )


def aggregate_cnn_legacy(
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


def update_global_linear_layer(
    global_layer, local_layer, pruned_indices_dict, scaler=1.0
):
    """Update the global layer inplacely with the weights and biases of the local layer.

    Args:
        global_layer: The global layer to be updated.
        local_layer: The local layer which is pruned from the global layer, and then updated.
        pruned_indices_dict: A dictionary containing the indices indicating which neurons have been pruned from global layer.
        scaler: A scalar to scale the weights of the local layer before updating the global layer.
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
    ] = local_layer.weight.detach().clone() * scaler

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
        ] = local_layer.bias.detach().clone() * scaler
    else:
        new_bias = None

    # Copy the updated weights and biases back to the global layer
    global_layer.weight.data.copy_(new_weight)
    if new_bias is not None:
        global_layer.bias.data.copy_(new_bias)


def update_global_conv_layer(
    global_layer, local_layer, pruned_indices_dict, scaler=1.0
):
    """Update the global layer inplacely with the weights and biases of the local layer.

    Args:
        global_layer: The global layer to be updated.
        local_layer: The local layer which is pruned from the global layer, and then updated.
        pruned_indices_dict: A dictionary containing the indices indicating which neurons have been pruned from global layer.
        scaler: A scalar to scale the weights of the local layer before updating the global layer.
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
    ] = local_layer.weight.detach().clone() * scaler

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
        ] = local_layer.bias.detach().clone() * scaler
    else:
        new_bias = None

    # Copy the updated weights and biases back to the global layer
    global_layer.weight.data.copy_(new_weight)
    if new_bias is not None:
        global_layer.bias.data.copy_(new_bias)
