import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet
from tqdm import tqdm


def calculate_model_size(model, print_result=True, unit="MB"):
    total_params = 0
    for param in model.parameters():
        # Multiply the size of each dimension to get the total number of elements
        total_params += param.numel()

    # Assuming each parameter is a 32-bit float
    memory_bytes = total_params * 4
    memory_kilobytes = memory_bytes / 1024
    memory_megabytes = memory_kilobytes / 1024

    if print_result:
        print(f"Total parameters: {total_params}")
        print(
            f"Memory Usage: {memory_bytes} bytes ({memory_kilobytes:.2f} KB / {memory_megabytes:.2f} MB)"
        )

    if unit == "KB":
        return memory_kilobytes
    elif unit == "MB":
        return memory_megabytes
    else:
        return memory_bytes


# Training function
def train(
    model,
    optimizer,
    criterion,
    train_loader,
    device="cuda",
    epochs=10,
    print_log=False,
):
    original_model_device = next(model.parameters()).device
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in tqdm(range(epochs), leave=False, desc="Training Epochs"):
        training_loss = 0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(train_loader)

        if print_log:
            correct = 0
            with torch.no_grad():
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            print(
                f"Train Epoch: {epoch}/{epochs}\tAverage Training Loss: {training_loss:.6f}\tAccuracy: {correct}/{len(train_loader.dataset)} ({100. * correct / len(train_loader.dataset):.0f}%)"
            )

    # Calculate the final training loss
    train_loss = 0.0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
        train_loss /= len(train_loader)

    model.to(original_model_device)
    return train_loss


def train_and_validate(
    model, train_loader, val_loader, optimizer, criterion, epochs=30, device="cuda"
):
    original_device = next(model.parameters()).device
    model.to(device)
    model.train()

    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_acc = correct / len(train_loader.dataset)
        # val_loss, val_acc, _ = test(model, device, val_loader, criterion)
        val_result = evaluate_acc(model, val_loader, device=device)
        print(
            f"Epoch {epoch + 1}/{epochs}, Traing Loss: {total_loss:.6f}, Training Accuracy: {correct}/{len(train_loader.dataset)} ({100. * correct / len(train_loader.dataset):.2f}%)\tValidation Loss: {val_result["loss"]:.6f}, Validation Accuracy: {val_result["accuracy"]:.4f}"
        )

        train_acc_list.append(train_acc)
        val_acc_list.append(val_result["accuracy"])

    model.to(original_device)
    return train_acc_list, val_acc_list


# Testing function
def test(model, criterion, test_loader, device="cuda", num_classes=10):
    original_model_device = next(model.parameters()).device
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    class_accuracy = {
        cls: class_correct[cls] / class_total[cls] for cls in sorted(class_total)
    }
    model.to(original_model_device)
    return test_loss, accuracy, class_accuracy


def evaluate_acc(model, dataloader, device="cuda", class_wise=False):
    original_model_device = next(model.parameters()).device
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    if class_wise:
        class_correct = dict()
        class_total = dict()

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if class_wise:
                for label, prediction in zip(target.view_as(pred), pred):
                    if label.item() not in class_correct:
                        class_correct[label.item()] = 0
                    if label.item() not in class_total:
                        class_total[label.item()] = 0

                    if label == prediction:
                        class_correct[label.item()] += 1

                    class_total[label.item()] += 1

    test_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    if class_wise:
        class_accuracy = {
            cls: class_correct[cls] / class_total[cls]
            for cls in sorted(class_total)
            if class_total[cls] > 0
        }

    model.to(original_model_device)

    result = {
        "loss": test_loss,
        "accuracy": accuracy,
    }
    if class_wise:
        result["class_wise_accuracy"] = class_accuracy

    return result


def replace_bn_with_ln(model: nn.Module, affine=False):
    """
    Replace all BatchNorm layers in the model with LayerNorm layers.

    ResNet18 model has the following structure:
    conv1: torch.Size([128, 64, 16, 16])
    layer1.0.conv1: torch.Size([128, 64, 8, 8])
    layer1.0.conv2: torch.Size([128, 64, 8, 8])
    layer1.1.conv1: torch.Size([128, 64, 8, 8])
    layer1.1.conv2: torch.Size([128, 64, 8, 8])
    layer2.0.conv1: torch.Size([128, 128, 4, 4])
    layer2.0.conv2: torch.Size([128, 128, 4, 4])
    layer2.0.downsample.0: torch.Size([128, 128, 4, 4])
    layer2.1.conv1: torch.Size([128, 128, 4, 4])
    layer2.1.conv2: torch.Size([128, 128, 4, 4])
    layer3.0.conv1: torch.Size([128, 256, 2, 2])
    layer3.0.conv2: torch.Size([128, 256, 2, 2])
    layer3.0.downsample.0: torch.Size([128, 256, 2, 2])
    layer3.1.conv1: torch.Size([128, 256, 2, 2])
    layer3.1.conv2: torch.Size([128, 256, 2, 2])
    layer4.0.conv1: torch.Size([128, 512, 1, 1])
    layer4.0.conv2: torch.Size([128, 512, 1, 1])
    layer4.0.downsample.0: torch.Size([128, 512, 1, 1])
    layer4.1.conv1: torch.Size([128, 512, 1, 1])
    layer4.1.conv2: torch.Size([128, 512, 1, 1])
    """
    if not isinstance(model, ResNet):
        raise ValueError("Only ResNet18 is supported for now.")

    # Replace all BatchNorm layers with LayerNorm layers
    model.bn1 = nn.LayerNorm([64, 16, 16], elementwise_affine=affine)

    model.layer1[0].bn1 = nn.LayerNorm([64, 8, 8], elementwise_affine=affine)
    model.layer1[0].bn2 = nn.LayerNorm([64, 8, 8], elementwise_affine=affine)
    model.layer1[1].bn1 = nn.LayerNorm([64, 8, 8], elementwise_affine=affine)
    model.layer1[1].bn2 = nn.LayerNorm([64, 8, 8], elementwise_affine=affine)

    model.layer2[0].bn1 = nn.LayerNorm([128, 4, 4], elementwise_affine=affine)
    model.layer2[0].bn2 = nn.LayerNorm([128, 4, 4], elementwise_affine=affine)
    model.layer2[0].downsample[1] = nn.LayerNorm([128, 4, 4], elementwise_affine=affine)
    model.layer2[1].bn1 = nn.LayerNorm([128, 4, 4], elementwise_affine=affine)
    model.layer2[1].bn2 = nn.LayerNorm([128, 4, 4], elementwise_affine=affine)

    model.layer3[0].bn1 = nn.LayerNorm([256, 2, 2], elementwise_affine=affine)
    model.layer3[0].bn2 = nn.LayerNorm([256, 2, 2], elementwise_affine=affine)
    model.layer3[0].downsample[1] = nn.LayerNorm([256, 2, 2], elementwise_affine=affine)
    model.layer3[1].bn1 = nn.LayerNorm([256, 2, 2], elementwise_affine=affine)
    model.layer3[1].bn2 = nn.LayerNorm([256, 2, 2], elementwise_affine=affine)

    model.layer4[0].bn1 = nn.LayerNorm([512, 1, 1], elementwise_affine=affine)
    model.layer4[0].bn2 = nn.LayerNorm([512, 1, 1], elementwise_affine=affine)
    model.layer4[0].downsample[1] = nn.LayerNorm([512, 1, 1], elementwise_affine=affine)
    model.layer4[1].bn1 = nn.LayerNorm([512, 1, 1], elementwise_affine=affine)
    model.layer4[1].bn2 = nn.LayerNorm([512, 1, 1], elementwise_affine=affine)


def replace_bn_with_sbn(model: nn.Module):
    """
    Replace all BatchNorm layers in the model with static BatchNorm layers.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m = nn.BatchNorm2d(m.num_features, affine=False)


def measure_time(repeats: int = 1):
    """Decorator, measure the function time costs with mean and variance.

    Args:
        repeats (int, optional): Repeat times for measuring. Defaults to 10.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            timings = []
            for _ in range(repeats):
                start_time = time.time()
                func_res = func(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                timings.append(elapsed)
            np_times = np.array(timings)
            average_time = np.mean(np_times)
            variance = np.var(np_times)
            # logging.info(f"[{func.__name__}] Average time over {repeats} runs: {average_time:.6f} seconds")
            print(
                f"[{func.__name__}] Average time over {repeats} runs: {average_time:.6f} seconds"
            )
            if repeats > 1:
                # logging.info(f"[{func.__name__}] Variance of times: {variance:.6f}")
                print(f"[{func.__name__}] Variance of times: {variance:.6f}")
            return func_res

        return wrapper

    return decorator
