import numpy as np
import torch
import torch.nn as nn

from modules.models import CNN
from modules.utils import (
    aggregate_conv_layers,
    aggregate_linear_layers,
    prune_conv_layer,
    prune_linear_layer,
)

original_cnn = CNN()

dropout_rate = 0.5

layer1_output_channels_to_prune = int(
    original_cnn.layer1[0].out_channels * dropout_rate
)
layer1_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_cnn.layer1[0].out_channels),
        layer1_output_channels_to_prune,
        replace=False,
    )
)
layer1_pruned_indices = {"output": layer1_output_indices_to_prune}
pruned_layer1 = prune_conv_layer(original_cnn.layer1[0], layer1_pruned_indices)

layer2_output_channels_to_prune = int(
    original_cnn.layer2[0].out_channels * dropout_rate
)
layer2_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_cnn.layer2[0].out_channels),
        layer2_output_channels_to_prune,
        replace=False,
    )
)
layer2_pruned_indices = {
    "input": layer1_output_indices_to_prune,
    "output": layer2_output_indices_to_prune,
}
pruned_layer2 = prune_conv_layer(original_cnn.layer2[0], layer2_pruned_indices)

layer3_output_channels_to_prune = int(
    original_cnn.layer3[0].out_channels * dropout_rate
)
layer3_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_cnn.layer3[0].out_channels),
        layer3_output_channels_to_prune,
        replace=False,
    )
)
layer3_pruned_indices = {
    "input": layer2_output_indices_to_prune,
    "output": layer3_output_indices_to_prune,
}
pruned_layer3 = prune_conv_layer(original_cnn.layer3[0], layer3_pruned_indices)

fc_pruned_indices = []
for channel_index in layer3_output_indices_to_prune:
    start_index = channel_index * 4 * 4
    end_index = (channel_index + 1) * 4 * 4
    fc_pruned_indices.extend(list(range(start_index, end_index)))
fc_input_indices_to_prune = np.sort(fc_pruned_indices)
pruned_fc = prune_linear_layer(original_cnn.fc, {"input": fc_input_indices_to_prune})

pruned_cnn = CNN()
pruned_cnn.layer1[0] = pruned_layer1
pruned_cnn.layer1[1] = nn.BatchNorm2d(pruned_layer1.out_channels)
pruned_cnn.layer2[0] = pruned_layer2
pruned_cnn.layer2[1] = nn.BatchNorm2d(pruned_layer2.out_channels)
pruned_cnn.layer3[0] = pruned_layer3
pruned_cnn.layer3[1] = nn.BatchNorm2d(pruned_layer3.out_channels)
pruned_cnn.fc = pruned_fc

# Aggregation
original_cnn.layer1[0].weight.data = torch.zeros_like(
    original_cnn.layer1[0].weight.data, device=original_cnn.layer1[0].weight.device
)
original_cnn.layer1[0].bias.data = torch.zeros_like(
    original_cnn.layer1[0].bias.data, device=original_cnn.layer1[0].bias.device
)
original_cnn.layer2[0].weight.data = torch.zeros_like(
    original_cnn.layer2[0].weight.data, device=original_cnn.layer2[0].weight.device
)
original_cnn.layer2[0].bias.data = torch.zeros_like(
    original_cnn.layer2[0].bias.data, device=original_cnn.layer2[0].bias.device
)
original_cnn.layer3[0].weight.data = torch.zeros_like(
    original_cnn.layer3[0].weight.data, device=original_cnn.layer3[0].weight.device
)
original_cnn.layer3[0].bias.data = torch.zeros_like(
    original_cnn.layer3[0].bias.data, device=original_cnn.layer3[0].bias.device
)
original_cnn.fc.weight.data = torch.zeros_like(
    original_cnn.fc.weight.data, device=original_cnn.fc.weight.device
)
original_cnn.fc.bias.data = torch.zeros_like(
    original_cnn.fc.bias.data, device=original_cnn.fc.bias.device
)

aggregate_conv_layers(
    original_cnn.layer1[0], [pruned_cnn.layer1[0]], [layer1_pruned_indices], [1]
)
aggregate_conv_layers(
    original_cnn.layer2[0], [pruned_cnn.layer2[0]], [layer2_pruned_indices], [1]
)
aggregate_conv_layers(
    original_cnn.layer3[0], [pruned_cnn.layer3[0]], [layer3_pruned_indices], [1]
)
aggregate_linear_layers(
    original_cnn.fc, [pruned_cnn.fc], [{"input": fc_input_indices_to_prune}], [1]
)

x = torch.randn(2, 3, 32, 32)
pruned_cnn_out = pruned_cnn(x)
original_cnn_out = original_cnn(x)
assert torch.equal(
    pruned_cnn_out, original_cnn_out
), f"Pruned CNN output: \n{pruned_cnn_out}\nOriginal CNN output: \n{original_cnn_out}"
