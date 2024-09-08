import torch
from tqdm import tqdm

from modules.evaluation import evaluate_acc


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
