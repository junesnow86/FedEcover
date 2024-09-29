import torch
from tqdm import tqdm


def train(
    model,
    optimizer,
    criterion,
    dataloader,
    device="cuda",
    epochs=10,
    verbose=False,
    local_training_correction=False,
    update_directions=None,
    correction_coefficient=1.0,
):
    original_model_device = next(model.parameters()).device
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in tqdm(range(epochs), leave=False, desc="Training Epochs"):
        training_loss = 0
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            if local_training_correction and update_directions is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        name = name.replace(".0.weight", ".weight").replace(
                            ".0.bias", ".bias"
                        )
                        # if epoch == epochs - 1:
                        #     grad_norm = param.grad.norm().item()
                        #     update_direction_norm = update_directions[name].norm().item()
                        #     print(f"Param: {name}, Grad Norm: {grad_norm}, Update Direction Norm: {update_direction_norm}")
                        # param.grad -= update_directions[name].to(device) * correction_coefficient
                        # Scale update_directions to the same magnitude as the gradient
                        scaling_factor = param.grad.norm() / (update_directions[name].norm() + 1e-7)
                        scaled_update_direction = update_directions[name].to(device) * scaling_factor
                        param.grad = param.grad * (1 - correction_coefficient) - scaled_update_direction * correction_coefficient

            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(dataloader)

        if verbose:
            correct = 0
            with torch.no_grad():
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            print(
                f"Train Epoch: {epoch}/{epochs}\tAverage Training Loss: {training_loss:.6f}\tAccuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.0f}%)"
            )

    # Calculate the final training loss
    train_loss = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
        train_loss /= len(dataloader)

    model.to(original_model_device)
    return train_loss


def scaffold_train(
    model,
    optimizer,
    criterion,
    dataloader,
    c_global,
    c_client,
    device="cuda",
    epochs=10,
    verbose=False,
):
    original_model_device = next(model.parameters()).device
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for name, param in model.named_parameters():
        name = name.replace(".0.weight", ".weight").replace(".0.bias", ".bias")
        c_global[name] = c_global[name].to(device)
        c_client[name] = c_client[name].to(device)
    model.train()

    for epoch in tqdm(range(epochs), leave=False, desc="Training Epochs"):
        training_loss = 0
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Apply SCAFFOLD control variates
            with torch.no_grad():
                for name, param in model.named_parameters():
                    name = name.replace(".0.weight", ".weight").replace(
                        ".0.bias", ".bias"
                    )
                    # param.grad += c_global[name] - c_client[name]

                    # Standardize the control variates
                    grad_mean = param.grad.mean()
                    grad_std = param.grad.std()

                    c_global_scaled = (c_global[name] - c_global[name].mean()) / (
                        c_global[name].std() + 1e-7
                    ) * grad_std + grad_mean
                    c_client_scaled = (c_client[name] - c_client[name].mean()) / (
                        c_client[name].std() + 1e-7
                    ) * grad_std + grad_mean

                    param.grad += c_global_scaled - c_client_scaled

            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(dataloader)

        if verbose:
            correct = 0
            with torch.no_grad():
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            print(
                f"Train Epoch: {epoch}/{epochs}\tAverage Training Loss: {training_loss:.6f}\tAccuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.0f}%)"
            )

    # Calculate the final training loss
    train_loss = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
        train_loss /= len(dataloader)

    model.to(original_model_device)
    for name, param in model.named_parameters():
        name = name.replace(".0.weight", ".weight").replace(".0.bias", ".bias")
        c_global[name] = c_global[name].to(original_model_device)
        c_client[name] = c_client[name].to(original_model_device)
    return train_loss
