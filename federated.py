import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from modules.utils import federated_averaging


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
def model_train(model, X_train, y_train, X_val, y_val, n_epochs=100, batch_size=16):
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
    # Read data
    data = pd.read_csv("sonar.csv", header=None)
    X = data.iloc[:, 0:60]
    y = data.iloc[:, 60]

    # Label encode the target from string to integer
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    # 分割数据集为两个子集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    X1_train, X1_val, y1_train, y1_val = train_test_split(
        X1, y1, test_size=0.1, random_state=42
    )
    X2_train, X2_val, y2_train, y2_val = train_test_split(
        X2, y2, test_size=0.1, random_state=42
    )
    print(len(X))
    print(len(X1_train), len(X2_train))

    # Convert to 2D PyTorch tensors
    X1_train = torch.tensor(X1_train.values, dtype=torch.float32)
    y1_train = torch.tensor(y1_train, dtype=torch.float32).reshape(-1, 1)
    X1_val = torch.tensor(X1_val.values, dtype=torch.float32)
    y1_val = torch.tensor(y1_val, dtype=torch.float32).reshape(-1, 1)
    X2_train = torch.tensor(X2_train.values, dtype=torch.float32)
    y2_train = torch.tensor(y2_train, dtype=torch.float32).reshape(-1, 1)
    X2_val = torch.tensor(X2_val.values, dtype=torch.float32)
    y2_val = torch.tensor(y2_val, dtype=torch.float32).reshape(-1, 1)

    central_model = PrunedSonarModel(dropout_rate=0.9)
    initial_weights = central_model.state_dict()
    participant_model1 = PrunedSonarModel(dropout_rate=0.9)
    participant_model1.load_state_dict(initial_weights)
    participant_model2 = PrunedSonarModel(dropout_rate=0.9)
    participant_model2.load_state_dict(initial_weights)

    # Train participant models
    acc1 = model_train(participant_model1, X1_train, y1_train, X1_val, y1_val)
    acc2 = model_train(participant_model2, X2_train, y2_train, X2_val, y2_val)
    print("Participant 1 Accuracy: %.2f" % acc1)
    print("Participant 2 Accuracy: %.2f" % acc2)

    # Aggregate participant models
    model_weights = [participant_model1.state_dict(), participant_model2.state_dict()]
    num_samples = [len(X1_train), len(X2_train)]
    avg_weights = federated_averaging(model_weights, num_samples)

    # Load aggregated weights into central model and test
    central_model.load_state_dict(avg_weights)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    y_pred = central_model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc)
    print("Central Model Accuracy: %.2f" % acc)
