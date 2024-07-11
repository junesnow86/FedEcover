import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from modules.models import SonarModel
from modules.utils import aggregate_linear_layers, prune_linear_layer

original_model = SonarModel()

dropout_rate = 0.2

layer1_output_neurons_to_prune = int(original_model.layer1.out_features * dropout_rate)
layer1_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_model.layer1.out_features),
        layer1_output_neurons_to_prune,
        replace=False,
    )
)
layer1_pruned_indices = {"output": layer1_output_indices_to_prune}
pruned_layer1 = prune_linear_layer(original_model.layer1, layer1_pruned_indices)

layer2_output_neurons_to_prune = int(original_model.layer2.out_features * dropout_rate)
layer2_output_indices_to_prune = np.sort(
    np.random.choice(
        range(original_model.layer2.out_features),
        layer2_output_neurons_to_prune,
        replace=False,
    )
)
layer2_pruned_indices = {
    "input": layer1_output_indices_to_prune,
    "output": layer2_output_indices_to_prune,
}
pruned_layer2 = prune_linear_layer(original_model.layer2, layer2_pruned_indices)


layer3_pruned_indices = {
    "input": layer2_output_indices_to_prune,
}
pruned_layer3 = prune_linear_layer(
    original_model.output,
    layer3_pruned_indices,
)

pruned_model = SonarModel()
pruned_model.layer1 = pruned_layer1
pruned_model.layer2 = pruned_layer2
pruned_model.output = pruned_layer3


def model_train(model, X_train, y_train, n_epochs=300, batch_size=16):
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


def test(model, X_val, y_val):
    model.eval()
    y_pred = model(X_val)
    acc = (y_pred.round() == y_val).float().mean()
    acc = float(acc)
    return acc


if __name__ == "__main__":
    # Read data
    data = pd.read_csv("sonar.csv", header=None)
    X = data.iloc[:, 0:60]
    y = data.iloc[:, 60]

    # Label encode the target from string to integer
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    # Convert to 2D PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    train_indices = None
    val_indices = None
    for train, val in kfold.split(X, y):
        train_indices = train
        val_indices = val

    model_train(pruned_model, X[train], y[train])
    acc = test(pruned_model, X[val], y[val])
    print("Pruned Model Accuracy: %.2f" % acc)

    acc = test(original_model, X[val], y[val])
    print("Original Model Accuracy: %.2f" % acc)

    # Aggregation
    original_model.layer1.weight.data = torch.zeros_like(
        original_model.layer1.weight.data, device=original_model.layer1.weight.device
    )
    original_model.layer1.bias.data = torch.zeros_like(
        original_model.layer1.bias.data, device=original_model.layer1.bias.device
    )
    original_model.layer2.weight.data = torch.zeros_like(
        original_model.layer2.weight.data, device=original_model.layer2.weight.device
    )
    original_model.layer2.bias.data = torch.zeros_like(
        original_model.layer2.bias.data, device=original_model.layer2.bias.device
    )
    original_model.output.weight.data = torch.zeros_like(
        original_model.output.weight.data, device=original_model.output.weight.device
    )
    original_model.output.bias.data = torch.zeros_like(
        original_model.output.bias.data, device=original_model.output.bias.device
    )
    aggregate_linear_layers(
        original_model.layer1, [pruned_model.layer1], [layer1_pruned_indices], [1]
    )
    aggregate_linear_layers(
        original_model.layer2, [pruned_model.layer2], [layer2_pruned_indices], [1]
    )
    aggregate_linear_layers(
        original_model.output, [pruned_model.output], [layer3_pruned_indices], [1]
    )

    acc = test(original_model, X[val], y[val])
    print("Aggregated Model Accuracy: %.2f" % acc)
