import numpy as np
import torch
import torch.nn as nn

from modules.utils import (
    aggregate_conv_layers,
    aggregate_linear_layers,
    prune_conv_layer,
    prune_linear_layer,
)

original_conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
original_conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
original_fc = nn.Linear(8 * 4 * 4, 2)

dropout_rate = 0.5

layer1_output_channels_to_prune = int(original_conv1.out_channels * dropout_rate)
layer1_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_conv1.out_channels),
        layer1_output_channels_to_prune,
        replace=False,
    )
)
layer1_output_indices_to_keep = np.setdiff1d(
    range(original_conv1.out_channels), layer1_output_indices_to_prune
)
layer1_pruned_indices = {"output": layer1_output_indices_to_prune}
pruned_conv1 = prune_conv_layer(original_conv1, layer1_pruned_indices)
with torch.no_grad():
    new_weights = torch.randn_like(pruned_conv1.weight)
    pruned_conv1.weight.copy_(new_weights)

layer2_output_channels_to_prune = int(original_conv2.out_channels * dropout_rate)
layer2_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_conv2.out_channels),
        layer2_output_channels_to_prune,
        replace=False,
    )
)
layer2_pruned_indices = {
    "input": layer1_output_indices_to_prune,
    "output": layer2_output_indices_to_prune,
}
pruned_conv2 = prune_conv_layer(original_conv2, layer2_pruned_indices)
with torch.no_grad():
    new_weights = torch.randn_like(pruned_conv2.weight)
    pruned_conv2.weight.copy_(new_weights)

fc_pruned_indices = []
for channel_index in layer2_output_indices_to_prune:
    start_index = channel_index * 4 * 4
    end_index = (channel_index + 1) * 4 * 4
    fc_pruned_indices.extend(list(range(start_index, end_index)))
fc_input_indices_to_prune = np.sort(fc_pruned_indices)
pruned_fc = prune_linear_layer(original_fc, {"input": fc_input_indices_to_prune})
with torch.no_grad():
    new_weights = torch.randn_like(pruned_fc.weight)
    pruned_fc.weight.copy_(new_weights)


original_conv1.weight.data = torch.zeros_like(original_conv1.weight.data)
original_conv1.bias.data = torch.zeros_like(original_conv1.bias.data)
original_conv2.weight.data = torch.zeros_like(original_conv2.weight.data)
original_conv2.bias.data = torch.zeros_like(original_conv2.bias.data)
original_fc.weight.data = torch.zeros_like(original_fc.weight.data)
original_fc.bias.data = torch.zeros_like(original_fc.bias.data)

aggregate_conv_layers(original_conv1, [pruned_conv1], [layer1_pruned_indices], [1])
aggregate_conv_layers(original_conv2, [pruned_conv2], [layer2_pruned_indices], [1])
aggregate_linear_layers(
    original_fc, [pruned_fc], [{"input": fc_input_indices_to_prune}], [1]
)


x = torch.randn(128, 3, 16, 16)

conv1_out = original_conv1(x)
conv1_out = nn.BatchNorm2d(4)(conv1_out)
conv1_out = nn.ReLU()(conv1_out)
conv1_out = nn.MaxPool2d(kernel_size=2, stride=2)(conv1_out)
pruned_conv1_out = pruned_conv1(x)
pruned_conv1_out = nn.BatchNorm2d(2)(pruned_conv1_out)
pruned_conv1_out = nn.ReLU()(pruned_conv1_out)
pruned_conv1_out = nn.MaxPool2d(kernel_size=2, stride=2)(pruned_conv1_out)

conv2_out = original_conv2(conv1_out)
conv2_out = nn.BatchNorm2d(8)(conv2_out)
conv2_out = nn.ReLU()(conv2_out)
conv2_out = nn.MaxPool2d(kernel_size=2, stride=2)(conv2_out)
pruned_conv2_out = pruned_conv2(pruned_conv1_out)
pruned_conv2_out = nn.BatchNorm2d(4)(pruned_conv2_out)
pruned_conv2_out = nn.ReLU()(pruned_conv2_out)
pruned_conv2_out = nn.MaxPool2d(kernel_size=2, stride=2)(pruned_conv2_out)

original_fc_out = original_fc(conv2_out.view(conv2_out.size(0), -1))
pruned_fc_out = pruned_fc(pruned_conv2_out.view(pruned_conv2_out.size(0), -1))
assert torch.equal(
    original_fc_out, pruned_fc_out
), f"original_fc_out:\n{original_fc_out}\npruned_fc_out:\n{pruned_fc_out}"
