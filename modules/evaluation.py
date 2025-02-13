import torch
import torch.nn as nn
from tqdm import tqdm


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
        for data, target in tqdm(dataloader, total=len(dataloader), leave=False, desc="Evaluating"):
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
