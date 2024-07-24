import torch
from tqdm import tqdm


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
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if print_log:
            correct = 0
            with torch.no_grad():
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            avg_loss = total_loss / len(train_loader)
            print(
                f"Train Epoch: {epoch}/{epochs}\tAverage Loss: {avg_loss:.6f}\tAccuracy: {correct}/{len(train_loader.dataset)} ({100. * correct / len(train_loader.dataset):.0f}%)"
            )

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

    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    class_accuracy = {
        cls: class_correct[cls] / class_total[cls] for cls in sorted(class_total)
    }
    model.to(original_model_device)
    return test_loss, accuracy, class_accuracy
