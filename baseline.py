from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def prune_linear_layer(layer, p):
    """
    Prune a linear layer by randomly dropping a proportion p of input and output neurons.

    Parameters:
    - layer: The linear layer to prune (an instance of torch.nn.Linear).
    - p: The proportion of neurons to drop.

    Returns:
    - new_layer: The new linear layer with pruned neurons.
    - pruned_indices: A dictionary with keys 'input' and 'output', indicating the indices of pruned neurons.
    """
    # Calculate the number of neurons to keep
    input_features = layer.in_features
    output_features = layer.out_features
    input_neurons_to_keep = int(input_features * (1 - p))
    output_neurons_to_keep = int(output_features * (1 - p))

    # Randomly select the neurons to keep
    input_indices_to_keep = np.sort(
        np.random.choice(range(input_features), input_neurons_to_keep, replace=False)
    )
    output_indices_to_keep = np.sort(
        np.random.choice(range(output_features), output_neurons_to_keep, replace=False)
    )

    print(input_indices_to_keep)
    print(output_indices_to_keep)

    # Extract the weights and biases for the remaining neurons
    new_weight = layer.weight.data[output_indices_to_keep, :][:, input_indices_to_keep]
    new_bias = (
        layer.bias.data[output_indices_to_keep] if layer.bias is not None else None
    )

    # Create a new Linear layer with the pruned neurons
    new_layer = torch.nn.Linear(input_neurons_to_keep, output_neurons_to_keep)
    new_layer.weight = torch.nn.Parameter(new_weight)
    new_layer.bias = torch.nn.Parameter(new_bias) if new_bias is not None else None

    # Record the pruned indices
    pruned_indices = {
        "input": np.setdiff1d(range(input_features), input_indices_to_keep),
        "output": np.setdiff1d(range(output_features), output_indices_to_keep),
    }

    return new_layer, pruned_indices


def aggregate_linear_layers(
    global_layer, layers, pruned_indices: List[Dict[str, np.ndarray]]
):
    global_output_size = global_layer.weight.data.shape[0]
    global_input_size = global_layer.weight.data.shape[1]

    # 遍历每个裁剪后的线性层及其对应的裁剪索引
    for layer, indices in zip(layers, pruned_indices):
        # 获取当前层的权重
        layer_weights = layer.weight.data

        # 获取被裁剪的输出神经元的索引
        pruned_input_indices = indices["input"]
        pruned_output_indices = indices["output"]

        # 获取未被裁剪的输入神经元的索引
        unpruned_input_indices = np.setdiff1d(
            range(global_input_size), pruned_input_indices
        )
        unpruned_output_indices = np.setdiff1d(
            range(global_output_size), pruned_output_indices
        )

        # 创建索引映射表
        input_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_input_indices)
        }
        output_index_map = {
            original_idx: new_idx
            for new_idx, original_idx in enumerate(unpruned_output_indices)
        }

        # 对于每个被裁剪的输出神经元索引
        for out_idx_global in unpruned_output_indices:
            for in_idx_global in unpruned_input_indices:
                out_idx_layer = output_index_map[out_idx_global]
                in_idx_layer = input_index_map[in_idx_global]
                # 将当前层的权重加到global_layer的对应位置
                global_layer.weight.data[out_idx_global, in_idx_global] += (
                    layer_weights[out_idx_layer, in_idx_layer]
                )

        # 检查并聚合bias
        if layer.bias is not None:
            # 确保global_layer有bias
            if global_layer.bias is None:
                raise ValueError("Global layer does not have bias")

            layer_bias = layer.bias.data
            for out_idx_global in unpruned_output_indices:
                out_idx_layer = output_index_map[out_idx_global]
                global_layer.bias.data[out_idx_global] += layer_bias[out_idx_layer]


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


def print_named_parameters(model):
    for name, param in model.named_parameters():
        print(name, param.shape)


# Define PyTorch model, with dropout at hidden layers
class SonarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(60, 30)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output(x))
        return x


class PrunedSonarModel(nn.Module):
    def __init__(
        self,
        input_size=60,
        layer1_size=60,
        layer2_size=30,
        output_size=1,
        dropout_rate=0.2,
    ):
        super().__init__()
        # Calculate pruned sizes
        pruned_layer1_size = int(layer1_size * (1 - dropout_rate))
        pruned_layer2_size = int(layer2_size * (1 - dropout_rate))

        # Initialize layers with pruned sizes
        self.layer1 = nn.Linear(input_size, pruned_layer1_size)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(pruned_layer1_size, pruned_layer2_size)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(pruned_layer2_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


# Helper function to train the model and return the validation result
def model_train(model, X_train, y_train, X_val, y_val, n_epochs=300, batch_size=16):
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    batch_start = torch.arange(0, len(X_train), batch_size)

    model.train()
    for epoch in range(n_epochs):
        for start in batch_start:
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate accuracy after training
    model.eval()
    y_pred = model(X_val)
    acc = (y_pred.round() == y_val).float().mean()
    acc = float(acc)
    return acc


if __name__ == "__main__":
    layer = nn.Linear(10, 10)
    print(layer.bias.data)
    # FIXME: 需要按顺序裁剪，否则会出现索引错乱
    new_layer, pruned_indices = prune_linear_layer(layer, 0.5)
    print(pruned_indices)
    print(new_layer.bias.data)
    aggregate_linear_layers(layer, [new_layer], [pruned_indices])
    print(layer.bias.data)

    # # Read data
    # data = pd.read_csv("sonar.csv", header=None)
    # X = data.iloc[:, 0:60]
    # y = data.iloc[:, 60]

    # # Label encode the target from string to integer
    # encoder = LabelEncoder()
    # encoder.fit(y)
    # y = encoder.transform(y)

    # # Convert to 2D PyTorch tensors
    # X = torch.tensor(X.values, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # # run 10-fold cross validation
    # kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # accuracies = []
    # pruned_accuracies = []
    # for train, test in kfold.split(X, y):
    #     # create model, train, and get accuracy
    #     model = SonarModel()
    #     acc = model_train(model, X[train], y[train], X[test], y[test])
    #     print("Accuracy: %.2f" % acc)
    #     accuracies.append(acc)

    #     # create pruned model, train, and get accuracy
    #     pruned_model = PrunedSonarModel()
    #     acc = model_train(pruned_model, X[train], y[train], X[test], y[test])
    #     print("Pruned Accuracy: %.2f" % acc)
    #     pruned_accuracies.append(acc)

    # # evaluate the model
    # mean = np.mean(accuracies)
    # std = np.std(accuracies)
    # print("Baseline: %.2f%% (+/- %.2f%%)" % (mean * 100, std * 100))

    # # evaluate the pruned model
    # mean = np.mean(pruned_accuracies)
    # std = np.std(pruned_accuracies)
    # print("Pruned: %.2f%% (+/- %.2f%%)" % (mean * 100, std * 100))

    # model = SonarModel()
    # pruned_model = PrunedSonarModel()
    # calculate_model_size(model)
    # calculate_model_size(pruned_model)
    # print_named_parameters(model)
    # print_named_parameters(pruned_model)
