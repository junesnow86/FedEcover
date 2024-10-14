import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.models import resnet18

from modules.constants import NORMALIZATION_STATS
from modules.data import TinyImageNet
from modules.training import train
from modules.evaluation import evaluate_acc
from modules.models import custom_resnet18

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZATION_STATS["imagenet"]["mean"],
            std=NORMALIZATION_STATS["imagenet"]["std"],
        ),
    ]
)

train_dataset = TinyImageNet(
    root_dir="data/tiny-imagenet-200", split="train", transform=transform
)
val_dataset = TinyImageNet(
    root_dir="data/tiny-imagenet-200", split="val", transform=transform
)

print(len(train_dataset.label_to_idx))
print(len(val_dataset.label_to_idx))

for k, v in train_dataset.label_to_idx.items():
    assert v == val_dataset.label_to_idx[k]
print("Label to index mapping is the same")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# model = resnet18(num_classes=len(train_dataset.label_to_idx), weights=None)
model = custom_resnet18(num_classes=len(train_dataset.label_to_idx), weights=None, input_shape=[64, 64])

critertion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loss = train(model, optimizer, critertion, train_loader, epochs=100, verbose=True)

evaluation_results = evaluate_acc(model, val_loader)
print(f"Validation accuracy: {evaluation_results['accuracy']:.4f}")
