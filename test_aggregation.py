import numpy as np
import torch
import torch.nn as nn

# from modules.models import CNN
from modules.utils import (
    # aggregate_conv_layers,
    aggregate_linear_layers,
    # prune_conv_layer,
    prune_linear_layer,
)

original_layer1 = nn.Linear(2, 4)
original_layer2 = nn.Linear(4, 1)

dropout_rate = 0.5

layer1_output_neurons_to_prune = int(original_layer1.out_features * dropout_rate)
layer1_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_layer1.out_features),
        layer1_output_neurons_to_prune,
        replace=False,
    )
)
layer1_pruned_indices = {"output": layer1_output_indices_to_prune}
pruned_layer1, pruned_indices1 = prune_linear_layer(
    original_layer1, layer1_pruned_indices
)

pruned_indices2 = {"input": pruned_indices1["output"]}
pruned_layer2, pruned_indices2 = prune_linear_layer(
    original_layer2, pruned_indices=pruned_indices2
)

pruned_layers = nn.Sequential(pruned_layer1, pruned_layer2)

x = torch.randn(1, 2)
pruned_out = pruned_layers(x)

original_layer1.weight.data = torch.zeros_like(original_layer1.weight.data)
original_layer1.bias.data = torch.zeros_like(original_layer1.bias.data)
original_layer2.weight.data = torch.zeros_like(original_layer2.weight.data)
original_layer2.bias.data = torch.zeros_like(original_layer2.bias.data)
aggregate_linear_layers(original_layer1, [pruned_layer1], [pruned_indices1], [1])
aggregate_linear_layers(original_layer2, [pruned_layer2], [pruned_indices2], [1])
aggregated_layers = nn.Sequential(original_layer1, original_layer2)
aggregated_out = aggregated_layers(x)

print(original_layer1.weight.data)
print(pruned_layer1.weight.data)
print(original_layer1.bias.data)
print(pruned_layer1.bias.data)
print("-----")

print(original_layer2.weight.data)
print(pruned_layer2.weight.data)
print(original_layer2.bias.data)
print(pruned_layer2.bias.data)
print("-----")

print(pruned_out)
print(aggregated_out)

# local_linear2, pruned_indices2 = prune_linear_layer(global_linear, 0.5)
# aggregate_linear_layers(
#     global_linear,
#     [local_linear1, local_linear2],
#     [pruned_indices1, pruned_indices2],
#     [1, 1],
# )

# global_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
# local_conv1, pruned_indices1 = prune_conv_layer(global_conv, 0.5)
# local_conv2, pruned_indices2 = prune_conv_layer(global_conv, 0.5)
# print(local_conv1.weight.shape)
# aggregate_conv_layers(
#     global_conv, [local_conv1, local_conv2], [pruned_indices1, pruned_indices2], [1, 1]
# )

# model = CNN()
# print(model.layer1[0])
