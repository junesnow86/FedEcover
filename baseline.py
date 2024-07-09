import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


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


# run 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = []
pruned_accuracies = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    model = SonarModel()
    acc = model_train(model, X[train], y[train], X[test], y[test])
    print("Accuracy: %.2f" % acc)
    accuracies.append(acc)

    # create pruned model, train, and get accuracy
    pruned_model = PrunedSonarModel()
    acc = model_train(pruned_model, X[train], y[train], X[test], y[test])
    print("Pruned Accuracy: %.2f" % acc)
    pruned_accuracies.append(acc)

# evaluate the model
mean = np.mean(accuracies)
std = np.std(accuracies)
print("Baseline: %.2f%% (+/- %.2f%%)" % (mean * 100, std * 100))

# evaluate the pruned model
mean = np.mean(pruned_accuracies)
std = np.std(pruned_accuracies)
print("Pruned: %.2f%% (+/- %.2f%%)" % (mean * 100, std * 100))

model = SonarModel()
pruned_model = PrunedSonarModel()
calculate_model_size(model)
calculate_model_size(pruned_model)
