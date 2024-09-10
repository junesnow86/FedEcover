import copy
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from modules.models import DropoutScaling
from modules.models.transformer import (
    DecoderBlock,
    EncoderBlock,
    FeedForward,
    MultiHeadAttention,
)
from .pruned_indices_dicts import (
    BlockPrunedIndicesDict,
    LayerPrunedIndicesDict,
)


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


def prune_linear_layer(
    layer, layer_prune_indices_dict: Optional[LayerPrunedIndicesDict] = None
):
    """Prune a linear layer by using provided `layer_prune_indices_dict` to directly select neurons to drop.

    Args:
        - linear_layer: The linear layer to prune (an instance of torch.nn.Linear).
        - layer_prune_indices_dict: A dictionary with keys 'input' and 'output', indicating the indices of neurons to prune directly.

    Returns:
        - new_layer: The new linear layer with pruned neurons and smaller size.
    """
    assert isinstance(
        layer, torch.nn.Linear
    ), "Input layer must be an instance of torch.nn.Linear"

    in_features = layer.in_features
    out_features = layer.out_features

    if layer_prune_indices_dict is not None:
        # Note: Make sure the pruned indices are in relative order
        input_indices_keep = np.sort(
            np.setdiff1d(
                range(in_features), layer_prune_indices_dict.get("input", np.array([]))
            )
        )
        output_indices_keep = np.sort(
            np.setdiff1d(
                range(out_features), layer_prune_indices_dict.get("output", np.array([]))
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


def prune_conv_layer(layer, layer_prune_indices_dict: Optional[LayerPrunedIndicesDict] = None):
    """Prune a convolution layer by using provided `layer_prune_indices_dict` to directly select channels to drop.

    Args:
        - layer: The convolution layer to prune (an instance of torch.nn.Conv2d).
        - layer_prune_indices_dict: A dictionary with keys 'input' and 'output', indicating the indices of channels to prune directly.

    Returns:
        - new_layer: The new convolution layer with pruned channels.
    """
    assert isinstance(
        layer, torch.nn.Conv2d
    ), "Input layer must be an instance of torch.nn.Conv2d"

    in_channels = layer.in_channels
    out_channels = layer.out_channels

    if layer_prune_indices_dict is not None:
        # Note: Make sure the pruned indices are in relative order
        in_indices_keep = np.sort(
            np.setdiff1d(
                range(in_channels), layer_prune_indices_dict.get("input", np.array([]))
            )
        )
        out_indices_keep = np.sort(
            np.setdiff1d(
                range(out_channels), layer_prune_indices_dict.get("output", np.array([]))
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


def prune_embedding_layer(
    layer, layer_prune_indices_dict: Optional[LayerPrunedIndicesDict] = None
):
    """Prune an embedding layer by using provided `layer_prune_indices_dict` to directly select embeddings to drop.

    Args:
        - layer: The embedding layer to prune (an instance of torch.nn.Embedding).
        - layer_prune_indices_dict: A dictionary with keys 'input' and 'output', indicating the indices of embeddings to prune directly.

    Returns:
        - new_layer: The new embedding layer with pruned embeddings.
    """
    assert isinstance(
        layer, torch.nn.Embedding
    ), "Input layer must be an instance of torch.nn.Embedding"

    num_embeddings = layer.num_embeddings
    embedding_dim = layer.embedding_dim

    if layer_prune_indices_dict is not None:
        # Note: Make sure the pruned indices are in relative order
        out_indices_keep = np.sort(
            np.setdiff1d(
                range(embedding_dim),
                layer_prune_indices_dict.get("output", np.array([])),
            )
        )

        new_weight = torch.index_select(
            layer.weight.data, 1, torch.tensor(out_indices_keep)
        )
    else:
        out_indices_keep = np.arange(embedding_dim)
        new_weight = layer.weight.detach().clone()

    new_layer = torch.nn.Embedding(num_embeddings, len(out_indices_keep))
    with torch.no_grad():
        new_layer.weight.data.copy_(new_weight)

    return new_layer


def prune_multihead_attention_layer(
    layer,
    layer_pruned_indices_dict: Optional[LayerPrunedIndicesDict] = None,
    dropout_rate=0.0,
    scaling=True,
):
    num_heads = layer.num_heads

    if layer_pruned_indices_dict is not None:
        new_W_q = prune_linear_layer(layer.W_q, layer_pruned_indices_dict)
        new_W_k = prune_linear_layer(layer.W_k, layer_pruned_indices_dict)
        new_W_v = prune_linear_layer(layer.W_v, layer_pruned_indices_dict)

        W_o_pruned_indices_dict = {
            "input": layer_pruned_indices_dict.get("output", np.array([])),
            "output": layer_pruned_indices_dict.get("input", np.array([])),
        }
        new_W_o = prune_linear_layer(layer.W_o, W_o_pruned_indices_dict)

        if scaling:
            new_W_q = nn.Sequential(new_W_q, DropoutScaling(dropout_rate))
            new_W_k = nn.Sequential(new_W_k, DropoutScaling(dropout_rate))
            new_W_v = nn.Sequential(new_W_v, DropoutScaling(dropout_rate))
            new_W_o = nn.Sequential(new_W_o, DropoutScaling(dropout_rate))
        else:
            # Wrap the pruned linear layers in nn.Sequential for alignment with aggregation functions
            new_W_q = nn.Sequential(new_W_q)
            new_W_k = nn.Sequential(new_W_k)
            new_W_v = nn.Sequential(new_W_v)
            new_W_o = nn.Sequential(new_W_o)

        new_d_model = new_W_o[0].out_features
        new_layer = MultiHeadAttention(new_d_model, num_heads)
        setattr(new_layer, "W_q", new_W_q)
        setattr(new_layer, "W_k", new_W_k)
        setattr(new_layer, "W_v", new_W_v)
        setattr(new_layer, "W_o", new_W_o)
    else:
        new_layer = copy.deepcopy(layer)

    return new_layer


def prune_feedforward_layer(
    layer,
    layer_pruned_indices_dict: Optional[LayerPrunedIndicesDict] = None,
    dropout_rate=0.0,
    scaling=True,
):
    if layer_pruned_indices_dict is not None:
        new_fc1 = prune_linear_layer(layer.fc1, layer_pruned_indices_dict)
        new_fc2 = prune_linear_layer(
            layer.fc2,
            {
                "input": layer_pruned_indices_dict.get("output", np.array([])),
                "output": layer_pruned_indices_dict.get("input", np.array([])),
            },
        )
        if scaling:
            new_fc1 = nn.Sequential(new_fc1, DropoutScaling(dropout_rate))
            new_fc2 = nn.Sequential(new_fc2, DropoutScaling(dropout_rate))
        else:
            # Wrap the pruned linear layers in nn.Sequential for alignment with aggregation functions
            new_fc1 = nn.Sequential(new_fc1)
            new_fc2 = nn.Sequential(new_fc2)

        new_d_model = new_fc2[0].out_features
        new_d_ff = new_fc1[0].out_features
        new_layer = FeedForward(new_d_model, new_d_ff)
        setattr(new_layer, "fc1", new_fc1)
        setattr(new_layer, "fc2", new_fc2)
    else:
        new_layer = copy.deepcopy(layer)

    return new_layer


def prune_transformer_block(
    block: Union[EncoderBlock, DecoderBlock],
    block_pruned_indices_dict: Optional[BlockPrunedIndicesDict] = None,
    dropout_rate: float = 0.0,
    scaling: bool = True,
):
    if isinstance(block, EncoderBlock):
        is_encoder = True
    elif isinstance(block, DecoderBlock):
        is_encoder = False
    else:
        raise ValueError(
            "Input block must be an instance of EncoderBlock or DecoderBlock"
        )

    self_attention_pruned_indices_dict = block_pruned_indices_dict.get(
        "self_attn", None
    )
    feedforward_pruned_indices_dict = block_pruned_indices_dict.get("feedforward", None)
    if not is_encoder:
        cross_attention_pruned_indices_dict = block_pruned_indices_dict.get(
            "cross_attn", None
        )
    else:
        cross_attention_pruned_indices_dict = None

    num_heads = block.num_heads

    if self_attention_pruned_indices_dict is not None:
        new_self_attn = prune_multihead_attention_layer(
            block.self_attn, self_attention_pruned_indices_dict, dropout_rate, scaling
        )
    else:
        new_self_attn = copy.deepcopy(block.self_attn)

    if feedforward_pruned_indices_dict is not None:
        new_ffn = prune_feedforward_layer(
            block.ffn, feedforward_pruned_indices_dict, dropout_rate, scaling
        )
    else:
        new_ffn = copy.deepcopy(block.ffn)

    if cross_attention_pruned_indices_dict is not None:
        new_cross_attn = prune_multihead_attention_layer(
            block.cross_attn, cross_attention_pruned_indices_dict, dropout_rate, scaling
        )
    elif not is_encoder:
        new_cross_attn = copy.deepcopy(block.cross_attn)

    if is_encoder:
        assert (
            new_self_attn.d_model == new_ffn.d_model
        ), f"Dimension mismatch between attention({new_self_attn.d_model}) and feedforward layers({new_ffn.d_model})"
    else:
        assert (
            new_self_attn.d_model == new_cross_attn.d_model
        ), "Dimension mismatch between self-attention and cross-attention layers"
        assert (
            new_self_attn.d_model == new_ffn.d_model
        ), "Dimension mismatch between self-attention and feedforward layers"

    new_d_model = new_self_attn.d_model
    new_d_ff = new_ffn.d_ff
    if is_encoder:
        new_layer = EncoderBlock(new_d_model, num_heads, new_d_ff)
        setattr(new_layer, "self_attn", new_self_attn)
        setattr(new_layer, "ffn", new_ffn)
    else:
        new_layer = DecoderBlock(new_d_model, num_heads, new_d_ff)
        setattr(new_layer, "self_attn", new_self_attn)
        setattr(new_layer, "cross_attn", new_cross_attn)
        setattr(new_layer, "ffn", new_ffn)

    # Update LayerNorms
    new_layer.norm1 = nn.LayerNorm(new_d_model)
    new_layer.norm2 = nn.LayerNorm(new_d_model)
    if not is_encoder:
        new_layer.norm3 = nn.LayerNorm(new_d_model)

    return new_layer
