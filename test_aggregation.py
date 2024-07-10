import torch.nn as nn

from modules.models import CNN
from modules.utils import (
    aggregate_conv_layers,
    aggregate_linear_layers,
    prune_conv_layer,
    prune_linear_layer,
)

global_linear = nn.Linear(10, 10)
local_linear1, pruned_indices1 = prune_linear_layer(global_linear, 0.5)
local_linear2, pruned_indices2 = prune_linear_layer(global_linear, 0.5)
print(local_linear1.weight.shape)
aggregate_linear_layers(
    global_linear,
    [local_linear1, local_linear2],
    [pruned_indices1, pruned_indices2],
    [1, 1],
)

global_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
local_conv1, pruned_indices1 = prune_conv_layer(global_conv, 0.5)
local_conv2, pruned_indices2 = prune_conv_layer(global_conv, 0.5)
print(local_conv1.weight.shape)
aggregate_conv_layers(
    global_conv, [local_conv1, local_conv2], [pruned_indices1, pruned_indices2], [1, 1]
)

model = CNN()
print(model.layer1[0])
