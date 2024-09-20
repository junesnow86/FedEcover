import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from modules.args_parser import get_args
from modules.constants import NORMALIZATION_STATS
from modules.evaluation import evaluate_acc
from modules.models import CNN, Ensemble
from modules.server import ServerRDBagging
from modules.training import train

# <======================================== Parse arguments ========================================>
args = get_args()
SAVE_DIR = args.save_dir
MODEL_TYPE = args.model
DATASET = args.dataset
if DATASET == "cifar10":
    NUM_CLASSES = 10
elif DATASET == "cifar100":
    NUM_CLASSES = 100
else:
    raise ValueError(f"Dataset {DATASET} not supported.")
ROUNDS = args.rounds
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size
NUM_CLIENTS = args.num_clients
AGG_WAY = args.aggregation
DEBUGGING = args.debugging
METHOD = args.method
SELECT_RATIO = args.select_ratio
LOCAL_VALIDATION_FREQUENCY = int(1 / args.select_ratio)
LOCAL_TRAIN_RATIO = args.local_train_ratio


# Set random seed for reproducibility
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# <======================================== Data preparation ========================================>
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            NORMALIZATION_STATS[DATASET]["mean"], NORMALIZATION_STATS[DATASET]["std"]
        ),
    ]
)
if DATASET == "cifar10":
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform
    )
elif DATASET == "cifar100":
    train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=False, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=False, transform=transform
    )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# <======================================== Model preparation ========================================>
if MODEL_TYPE == "cnn":
    model = CNN(num_classes=NUM_CLASSES)
elif MODEL_TYPE == "resnet":
    model = resnet18(num_classes=NUM_CLASSES, weights=None)
else:
    raise ValueError(f"Model {MODEL_TYPE} not supported.")
print(f"[Model Architecture]\n{model}")

server = ServerRDBagging(
    global_model=model,
    num_clients=NUM_CLIENTS,
    client_capacities=[0.5] * NUM_CLIENTS,
    model_out_dim=NUM_CLASSES,
    select_ratio=SELECT_RATIO,
    scaling=True,
    strategy="p-based-frequent",
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# <======================================== Training ========================================>
for round in range(ROUNDS):
    print(f"Round {round + 1}/{ROUNDS}")
    train_loss = train(model, optimizer, criterion, train_loader, device, EPOCHS)
    val_result = evaluate_acc(model, test_loader, device)
    print(
        f"Train Loss: {train_loss:.4f}\tValidation Loss: {val_result['loss']:.4f}\tValidation Accuracy: {val_result['accuracy']:.4f}"
    )

    _, _, _, submodels = server.distribute()
    ensemble = Ensemble(submodels)
    val_result = evaluate_acc(ensemble, test_loader, device)
    print(
        f"{len(submodels)} models Ensemble\tValidation Loss: {val_result['loss']:.4f}\tValidation Accuracy: {val_result['accuracy']:.4f}"
    )
    print("=" * 100)
