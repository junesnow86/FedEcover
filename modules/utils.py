import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from modules.models import CNN


def prune_linear_layer(linear_layer, pruned_indices=None):
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
        input_indices_to_keep = np.setdiff1d(
            range(input_features), pruned_indices.get("input", np.array([]))
        )
        output_indices_to_keep = np.setdiff1d(
            range(output_features), pruned_indices.get("output", np.array([]))
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


def prune_conv_layer(conv_layer, pruned_indices=None):
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
        in_indices_to_keep = np.setdiff1d(
            range(in_channels), pruned_indices.get("input", np.array([]))
        )
        out_indices_to_keep = np.setdiff1d(
            range(out_channels), pruned_indices.get("output", np.array([]))
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


def aggregate_linear_layers(
    global_linear_layer,
    linear_layer_list: List[torch.nn.Linear],
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
    weight_sample_accumulator = torch.zeros((global_output_size, global_input_size))
    if global_linear_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_linear_layer.bias.data)
        bias_sample_accumulator = torch.zeros(global_output_size)

    for linear_layer, pruned_indices, num_samples in zip(
        linear_layer_list, pruned_indices_list, num_samples_list
    ):
        layer_weights = linear_layer.weight.data

        unpruned_input_indices = np.setdiff1d(
            range(global_input_size), pruned_indices.get("input", np.array([]))
        )
        unpruned_output_indices = np.setdiff1d(
            range(global_output_size), pruned_indices.get("output", np.array([]))
        )

        input_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_input_indices)
        }
        output_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_output_indices)
        }

        for out_idx_global in unpruned_output_indices:
            for in_idx_global in unpruned_input_indices:
                out_idx_layer = output_index_map[out_idx_global]
                in_idx_layer = input_index_map[in_idx_global]
                weight_accumulator[out_idx_global, in_idx_global] += (
                    layer_weights[out_idx_layer, in_idx_layer] * num_samples
                )
                weight_sample_accumulator[out_idx_global, in_idx_global] += num_samples

        if linear_layer.bias is not None:
            layer_bias = linear_layer.bias.data
            for out_idx_global in unpruned_output_indices:
                out_idx_layer = output_index_map[out_idx_global]
                bias_accumulator[out_idx_global] += (
                    layer_bias[out_idx_layer] * num_samples
                )
                bias_sample_accumulator[out_idx_global] += num_samples

        # Normalize the accumulated weights and biases by the number of samples
        for out_idx_global in unpruned_output_indices:
            for in_idx_global in unpruned_input_indices:
                if weight_sample_accumulator[out_idx_global, in_idx_global] > 0:
                    global_linear_layer.weight.data[out_idx_global, in_idx_global] = (
                        weight_accumulator[out_idx_global, in_idx_global]
                        / weight_sample_accumulator[out_idx_global, in_idx_global]
                    )

        if global_linear_layer.bias is not None:
            for out_idx_global in unpruned_output_indices:
                if bias_sample_accumulator[out_idx_global] > 0:
                    global_linear_layer.bias.data[out_idx_global] = (
                        bias_accumulator[out_idx_global]
                        / bias_sample_accumulator[out_idx_global]
                    )


def aggregate_conv_layers(
    global_conv_layer,
    conv_layer_list: List[torch.nn.Conv2d],
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
    weight_sample_accumulator = torch.zeros((global_out_channels, global_in_channels))
    if global_conv_layer.bias is not None:
        bias_accumulator = torch.zeros_like(global_conv_layer.bias.data)
        bias_sample_accumulator = torch.zeros(global_out_channels)

    for conv_layer, pruned_indices, num_samples in zip(
        conv_layer_list, pruned_indices_list, num_samples_list
    ):
        layer_weights = conv_layer.weight.data

        unpruned_in_indices = np.setdiff1d(
            range(global_in_channels), pruned_indices.get("input", np.array([]))
        )
        unpruned_out_indices = np.setdiff1d(
            range(global_out_channels), pruned_indices.get("output", np.array([]))
        )

        input_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_in_indices)
        }
        output_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_out_indices)
        }

        for out_idx_global in unpruned_out_indices:
            for in_idx_global in unpruned_in_indices:
                out_idx_layer = output_index_map[out_idx_global]
                in_idx_layer = input_index_map[in_idx_global]
                weight_accumulator[out_idx_global, in_idx_global, :, :] += (
                    layer_weights[out_idx_layer, in_idx_layer, :, :] * num_samples
                )
                weight_sample_accumulator[out_idx_global, in_idx_global] += num_samples

        if conv_layer.bias is not None:
            layer_bias = conv_layer.bias.data
            for out_idx_global in unpruned_out_indices:
                out_idx_layer = output_index_map[out_idx_global]
                bias_accumulator[out_idx_global] += (
                    layer_bias[out_idx_layer] * num_samples
                )
                bias_sample_accumulator[out_idx_global] += num_samples

        # Normalize the accumulated weights and biases by the number of samples
        for out_idx_global in unpruned_out_indices:
            for in_idx_global in unpruned_in_indices:
                if weight_sample_accumulator[out_idx_global, in_idx_global] > 0:
                    global_conv_layer.weight.data[
                        out_idx_global, in_idx_global, :, :
                    ] = (
                        weight_accumulator[out_idx_global, in_idx_global, :, :]
                        / weight_sample_accumulator[out_idx_global, in_idx_global]
                    )

        if global_conv_layer.bias is not None:
            for out_idx_global in unpruned_out_indices:
                if bias_sample_accumulator[out_idx_global].sum() > 0:
                    global_conv_layer.bias.data[out_idx_global] = (
                        bias_accumulator[out_idx_global]
                        / bias_sample_accumulator[out_idx_global]
                    )


def prune_cnn(original_cnn: CNN, dropout_rate=0.5, **indices_to_prune):
    indices_to_prune_conv1 = indices_to_prune.get("indices_to_prune_conv1", None)
    indices_to_prune_conv2 = indices_to_prune.get("indices_to_prune_conv2", None)
    indices_to_prune_conv3 = indices_to_prune.get("indices_to_prune_conv3", None)
    indices_to_prune_fc = indices_to_prune.get("indices_to_prune_fc", None)

    conv1 = copy.deepcopy(original_cnn.layer1[0])
    if indices_to_prune_conv1 is None:
        num_output_channels_to_prune_conv1 = int(conv1.out_channels * dropout_rate)
        output_indices_to_prune_conv1 = np.random.choice(
            conv1.out_channels, num_output_channels_to_prune_conv1, replace=False
        )
        indices_to_prune_conv1 = {"output": output_indices_to_prune_conv1}
    pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)

    conv2 = copy.deepcopy(original_cnn.layer2[0])
    if indices_to_prune_conv2 is None:
        num_output_channels_to_prune_conv2 = int(conv2.out_channels * dropout_rate)
        output_indices_to_prune_conv2 = np.random.choice(
            conv2.out_channels, num_output_channels_to_prune_conv2, replace=False
        )
        indices_to_prune_conv2 = {
            "input": output_indices_to_prune_conv1,
            "output": output_indices_to_prune_conv2,
        }
    pruned_layer2 = prune_conv_layer(conv2, indices_to_prune_conv2)

    conv3 = copy.deepcopy(original_cnn.layer3[0])
    if indices_to_prune_conv3 is None:
        num_output_channels_to_prune_conv3 = int(conv3.out_channels * dropout_rate)
        output_indices_to_prune_conv3 = np.random.choice(
            conv3.out_channels, num_output_channels_to_prune_conv3, replace=False
        )
        indices_to_prune_conv3 = {
            "input": output_indices_to_prune_conv2,
            "output": output_indices_to_prune_conv3,
        }
    pruned_layer3 = prune_conv_layer(conv3, indices_to_prune_conv3)

    fc = copy.deepcopy(original_cnn.fc)
    if indices_to_prune_fc is None:
        input_indices_to_prune_fc = []
        for channel_index in output_indices_to_prune_conv3:
            start_index = channel_index * 4 * 4
            end_index = (channel_index + 1) * 4 * 4
            input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
        input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
        indices_to_prune_fc = {"input": input_indices_to_prune_fc}
    pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)

    scale_factor = 1 / (1 - dropout_rate)
    pruned_layer1.weight.data *= scale_factor
    pruned_layer2.weight.data *= scale_factor
    pruned_layer3.weight.data *= scale_factor
    # pruned_fc.weight.data *= scale_factor

    pruned_cnn = CNN()
    pruned_cnn.layer1[0] = pruned_layer1
    pruned_cnn.layer2[0] = pruned_layer2
    pruned_cnn.layer3[0] = pruned_layer3
    pruned_cnn.fc = pruned_fc

    return (
        pruned_cnn,
        indices_to_prune_conv1,
        indices_to_prune_conv2,
        indices_to_prune_conv3,
        indices_to_prune_fc,
    )


def create_random_even_groups(num_total_elements, num_groups):
    num_elements_per_group = num_total_elements // num_groups
    indices = np.arange(num_total_elements)
    np.random.shuffle(indices)
    indices = indices[: num_elements_per_group * num_groups]
    return np.array_split(indices, num_groups)


def select_random_group(groups):
    group_index = np.random.choice(len(groups))
    selected_group = groups.pop(group_index)
    return selected_group, groups


def prune_cnn_into_groups(
    original_cnn: CNN, dropout_rate=0.5
) -> Tuple[List[CNN], List[Dict]]:
    num_groups = max(int(1 / (1 - dropout_rate)), 1)
    pruned_models = []
    indices_to_prune_list = []

    conv1 = copy.deepcopy(original_cnn.layer1[0])
    conv2 = copy.deepcopy(original_cnn.layer2[0])
    conv3 = copy.deepcopy(original_cnn.layer3[0])
    fc = copy.deepcopy(original_cnn.fc)

    groups_conv1 = create_random_even_groups(conv1.out_channels, num_groups)
    groups_conv2 = create_random_even_groups(conv2.out_channels, num_groups)
    groups_conv3 = create_random_even_groups(conv3.out_channels, num_groups)

    for _ in range(num_groups):
        group_conv1, groups_conv1 = select_random_group(groups_conv1)
        group_conv2, groups_conv2 = select_random_group(groups_conv2)
        group_conv3, groups_conv3 = select_random_group(groups_conv3)

        indices_to_prune_conv1 = {"output": group_conv1}
        indices_to_prune_conv2 = {
            "input": group_conv1,
            "output": group_conv2,
        }
        indices_to_prune_conv3 = {
            "input": group_conv2,
            "output": group_conv3,
        }

        input_indices_to_prune_fc = []
        for channel_index in group_conv3:
            start_index = channel_index * 4 * 4
            end_index = (channel_index + 1) * 4 * 4
            input_indices_to_prune_fc.extend(list(range(start_index, end_index)))
        input_indices_to_prune_fc = np.sort(input_indices_to_prune_fc)
        indices_to_prune_fc = {"input": input_indices_to_prune_fc}

        pruned_layer1 = prune_conv_layer(conv1, indices_to_prune_conv1)
        pruned_layer2 = prune_conv_layer(conv2, indices_to_prune_conv2)
        pruned_layer3 = prune_conv_layer(conv3, indices_to_prune_conv3)
        pruned_fc = prune_linear_layer(fc, indices_to_prune_fc)

        scale_factor = 1 / (1 - dropout_rate)
        pruned_layer1.weight.data *= scale_factor
        pruned_layer2.weight.data *= scale_factor
        pruned_layer3.weight.data *= scale_factor
        # pruned_fc.weight.data *= scale_factor

        pruned_cnn = CNN()
        pruned_cnn.layer1[0] = pruned_layer1
        pruned_cnn.layer2[0] = pruned_layer2
        pruned_cnn.layer3[0] = pruned_layer3
        pruned_cnn.fc = pruned_fc

        pruned_models.append(pruned_cnn)
        indices_to_prune_list.append(
            {
                "indices_to_prune_conv1": indices_to_prune_conv1,
                "indices_to_prune_conv2": indices_to_prune_conv2,
                "indices_to_prune_conv3": indices_to_prune_conv3,
                "indices_to_prune_fc": indices_to_prune_fc,
            }
        )

    return pruned_models, indices_to_prune_list


def aggregate_cnn(
    original_cnn: CNN,
    pruned_cnn_list: List[CNN],
    num_samples_list,
    dropout_rate_list,
    indices_to_prune_dict: Dict[str, list],
):
    indices_to_prune_conv1 = indices_to_prune_dict["indices_to_prune_conv1"]
    indices_to_prune_conv2 = indices_to_prune_dict["indices_to_prune_conv2"]
    indices_to_prune_conv3 = indices_to_prune_dict["indices_to_prune_conv3"]
    indices_to_prune_fc = indices_to_prune_dict["indices_to_prune_fc"]

    for i in range(len(pruned_cnn_list)):
        scale_factor = 1 - dropout_rate_list[i]
        pruned_cnn_list[i].layer1[0].weight.data *= scale_factor
        pruned_cnn_list[i].layer2[0].weight.data *= scale_factor
        pruned_cnn_list[i].layer3[0].weight.data *= scale_factor
        # pruned_cnn_list[i].fc.weight.data *= scale_factor

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


def vanilla_federated_averaging(model_weights, sample_numbers):
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

    return avg_weights


def calculate_model_size(model):
    total_params = 0
    for param in model.parameters():
        # Multiply the size of each dimension to get the total number of elements
        total_params += param.numel()

    # Assuming each parameter is a 32-bit float
    memory_bytes = total_params * 4
    memory_kilobytes = memory_bytes / 1024
    memory_megabytes = memory_kilobytes / 1024

    print(f"Total parameters: {total_params}")
    print(
        f"Memory Usage: {memory_bytes} bytes ({memory_kilobytes:.2f} KB / {memory_megabytes:.2f} MB)"
    )


# Training function
def train(
    model, device, train_loader, optimizer, criterion, epochs=30, print_log=False
):
    original_model_device = next(model.parameters()).device
    model.to(device)
    model.train()

    for epoch in tqdm(range(epochs), leave=False):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if print_log:
            avg_loss = total_loss / len(train_loader)
            print(f"Train Epoch: {epoch}/{epochs} \tAverage Loss: {avg_loss:.6f}")

    model.to(original_model_device)


# Testing function
def test(model, device, test_loader, criterion, num_classes=10):
    original_model_device = next(model.parameters()).device
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = dict()
    class_total = dict()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            for label, prediction in zip(target.view_as(pred), pred):
                if label.item() not in class_correct:
                    class_correct[label.item()] = 0
                if label.item() not in class_total:
                    class_total[label.item()] = 0

                if label == prediction:
                    class_correct[label.item()] += 1

                class_total[label.item()] += 1

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    class_accuracy = {
        cls: class_correct[cls] / class_total[cls] for cls in sorted(class_total)
    }
    model.to(original_model_device)
    return test_loss, accuracy, class_accuracy
