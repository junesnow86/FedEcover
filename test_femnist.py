import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.data import FEMNIST
from modules.evaluation import evaluate_acc
from modules.models import FEMNISTCNN
from modules.training import train

model = FEMNISTCNN("femnist")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

test_data = FEMNIST("data", train=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

trained_user = 0
for user in os.listdir("data/femnist/train"):
    train_data = FEMNIST("data", train=True, user_id=user.replace(".json", ""))
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    train(model, optimizer, criterion, train_loader, epochs=1)
    trained_user += 1

    if trained_user % 100 == 0:
        print(f"Trained {trained_user} users")
        eval_result = evaluate_acc(model, test_loader)
        print(f"Test accuracy: {eval_result['accuracy']:.4f}")
