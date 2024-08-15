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
        val_loss, val_acc, _ = test(model, device, val_loader, criterion)
        print(
            f"Epoch {epoch + 1}/{epochs}, Traing Loss: {total_loss:.6f}, Training Accuracy: {correct}/{len(train_loader.dataset)} ({100. * correct / len(train_loader.dataset):.2f}%)\tValidation Loss: {val_loss:.6f}, Validation Accuracy: {val_acc}"
        )

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

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
