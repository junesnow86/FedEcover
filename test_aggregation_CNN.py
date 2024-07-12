import numpy as np
import torch
import torch.nn as nn

from modules.utils import (
    aggregate_conv_layers,
    aggregate_linear_layers,
    prune_conv_layer,
    prune_linear_layer,
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256 * 4 * 4, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


original_cnn = CNN()

dropout_rate = 0.5

conv1 = original_cnn.layer1[0]
num_output_channels_to_prune_conv1 = int(conv1.out_channels * dropout_rate)
output_indices_to_prune_conv1 = np.random.choice(
    conv1.out_channels, num_output_channels_to_prune_conv1, replace=False
)
indices_to_prune_conv1 = {"output": output_indices_to_prune_conv1}
pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)

conv2 = original_cnn.layer2[0]
num_output_channels_to_prune_conv2 = int(conv2.out_channels * dropout_rate)
output_indices_to_prune_conv2 = np.random.choice(
    conv2.out_channels, num_output_channels_to_prune_conv2, replace=False
)
indices_to_prune_conv2 = {
    "input": output_indices_to_prune_conv1,
    "output": output_indices_to_prune_conv2,
}
pruned_layer2 = prune_conv_layer(conv2, indices_to_prune_conv2)

conv3 = original_cnn.layer3[0]
num_output_channels_to_prune_conv3 = int(conv3.out_channels * dropout_rate)
output_indices_to_prune_conv3 = np.random.choice(
    conv3.out_channels, num_output_channels_to_prune_conv3, replace=False
)
indices_to_prune_conv3 = {
    "input": output_indices_to_prune_conv2,
    "output": output_indices_to_prune_conv3,
}
pruned_layer3 = prune_conv_layer(conv3, indices_to_prune_conv3)

fc = original_cnn.fc
input_indices_to_prune_fc = []
for channel_index in output_indices_to_prune_conv3:
    start_index = channel_index * 4 * 4
    end_index = (channel_index + 1) * 4 * 4
    input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
indices_to_prune_fc = {"input": input_indices_to_prune_fc}
pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)

pruned_cnn = CNN()
pruned_cnn.layer1[0] = pruned_layer1
# pruned_cnn.layer1[1] = nn.BatchNorm2d(pruned_layer1.out_channels)
pruned_cnn.layer2[0] = pruned_layer2
# pruned_cnn.layer2[1] = nn.BatchNorm2d(pruned_layer2.out_channels)
pruned_cnn.layer3[0] = pruned_layer3
# pruned_cnn.layer3[1] = nn.BatchNorm2d(pruned_layer3.out_channels)
pruned_cnn.fc = pruned_fc

# Make all the parameters of the original CNN zero
# original_cnn.layer1[0].weight.data = torch.zeros_like(
#     original_cnn.layer1[0].weight.data, device=original_cnn.layer1[0].weight.device
# )
# original_cnn.layer1[0].bias.data = torch.zeros_like(
#     original_cnn.layer1[0].bias.data, device=original_cnn.layer1[0].bias.device
# )
# original_cnn.layer2[0].weight.data = torch.zeros_like(
#     original_cnn.layer2[0].weight.data, device=original_cnn.layer2[0].weight.device
# )
# original_cnn.layer2[0].bias.data = torch.zeros_like(
#     original_cnn.layer2[0].bias.data, device=original_cnn.layer2[0].bias.device
# )
# original_cnn.layer3[0].weight.data = torch.zeros_like(
#     original_cnn.layer3[0].weight.data, device=original_cnn.layer3[0].weight.device
# )
# original_cnn.layer3[0].bias.data = torch.zeros_like(
#     original_cnn.layer3[0].bias.data, device=original_cnn.layer3[0].bias.device
# )
# original_cnn.fc.weight.data = torch.zeros_like(
#     original_cnn.fc.weight.data, device=original_cnn.fc.weight.device
# )
# original_cnn.fc.bias.data = torch.zeros_like(
#     original_cnn.fc.bias.data, device=original_cnn.fc.bias.device
# )
for param in original_cnn.parameters():
    param.data = torch.zeros_like(param.data)

# Aggregation
aggregate_conv_layers(
    original_cnn.layer1[0], [pruned_cnn.layer1[0]], [indices_to_prune_conv1], [2]
)
aggregate_conv_layers(
    original_cnn.layer2[0], [pruned_cnn.layer2[0]], [indices_to_prune_conv2], [2]
)
aggregate_conv_layers(
    original_cnn.layer3[0], [pruned_cnn.layer3[0]], [indices_to_prune_conv3], [2]
)
aggregate_linear_layers(original_cnn.fc, [pruned_cnn.fc], [indices_to_prune_fc], [2])

x = torch.randn(2, 3, 32, 32)
pruned_cnn_out = pruned_cnn(x)
original_cnn_out = original_cnn(x)
assert torch.equal(
    pruned_cnn_out, original_cnn_out
), f"Pruned CNN output: \n{pruned_cnn_out}\nOriginal CNN output: \n{original_cnn_out}"
