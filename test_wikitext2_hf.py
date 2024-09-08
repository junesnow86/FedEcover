import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from modules.data import NextWordPredictionDataset
from modules.models import Transformer
from modules.pruning import prune_transformer
from modules.aggregation import aggregate_transformer
from modules.utils import calculate_model_size

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset["train"]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Preprocess data
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    return tokenized


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)


train_dataset = NextWordPredictionDataset(tokenized_dataset["train"])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = NextWordPredictionDataset(tokenized_dataset["validation"])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

global_model = Transformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    num_heads=8,
    num_layers=3,
    d_model=512,
    d_ff=2048,
    max_seq_len=128,
)

print(f"[Global Model Architecture]\n{global_model}")
print(f"Original model size: {calculate_model_size(global_model, print_result=False)} MB")


dropout_rate = 0.01
pruned_model, model_pruned_indices_dict = prune_transformer(global_model, dropout_rate, scaling=True)
print(f"[Model Architecture]\n{pruned_model}")
print(f"Pruned model size: {calculate_model_size(pruned_model, print_result=False)} MB")

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(pruned_model.parameters(), lr=0.0001)


# Training loop
def train(model, dataloader, criterion, optimizer, device, print_interval=100):
    original_device = next(model.parameters()).device
    model.to(device)

    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for batch_idx, batch in tqdm(enumerate(dataloader), desc="Training"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, input_ids)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        mask = labels != -100
        correct_predictions += (predictions[mask] == labels[mask]).sum().item()
        total_predictions += mask.sum().item()

        if (batch_idx + 1) % print_interval == 0:
            accuracy = correct_predictions / total_predictions
            print(
                f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}"
            )

    model.to(original_device)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    original_device = next(model.parameters()).device
    model.to(device)
    
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            total_loss += loss.item()

            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            mask = labels != -100
            correct_predictions += (predictions[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()

    model.to(original_device)

    accuracy = correct_predictions / total_predictions
    return total_loss / len(dataloader), accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 1
for epoch in range(num_epochs):
    train_loss, train_acc = train(pruned_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(pruned_model, val_loader, criterion, device)
    print(
        f"Epoch {epoch + 1}/{num_epochs}\tTrain Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}"
    )

val_loss, val_acc = evaluate(global_model, val_loader, criterion, device)
print(f"Global Model Val Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}")

val_loss, val_acc = evaluate(pruned_model, val_loader, criterion, device)
print(f"Pruned Model Val Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}")

aggregate_transformer(
    global_model=global_model,
    local_models=[pruned_model],
    model_pruned_indices_dicts=[model_pruned_indices_dict],
    client_weights=[2801],
)
val_loss, val_acc = evaluate(global_model, val_loader, criterion, device)
print(f"Aggregated Model Val Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}")
