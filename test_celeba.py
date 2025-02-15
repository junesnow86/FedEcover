import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.data import CelebA
from modules.evaluation import evaluate_acc
from modules.models import CNN
from modules.training import train

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

model = CNN("celeba")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

test_data = CelebA("data", train=False, augmentation=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

trained_user = 0
for user in os.listdir("data/celeba/train/by_user"):
    train_data = CelebA("data", train=True, user_id=user, augmentation=True)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    train(model, optimizer, criterion, train_loader, epochs=1)
    trained_user += 1

    if trained_user % 100 == 0:
        print(f"Trained {trained_user} users")
        eval_result = evaluate_acc(model, test_loader)
        print(f"Test accuracy: {eval_result['accuracy']:.4f}")
