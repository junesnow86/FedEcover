import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from modules.dataset import NextWordPredictionDataset
from modules.models import Transformer

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
        max_length=128,
    )
    return tokenized


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)


train_dataset = NextWordPredictionDataset(tokenized_dataset["train"])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = NextWordPredictionDataset(tokenized_dataset["validation"])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = Transformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    num_heads=8,
    num_layers=3,
    d_model=256,
    d_ff=1024,
    max_seq_len=128,
)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)


# Training loop
def train(model, dataloader, criterion, optimizer, device, print_interval=100):
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

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
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

    accuracy = correct_predictions / total_predictions
    return total_loss / len(dataloader), accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(
        f"Epoch {epoch + 1}/{num_epochs}\tTrain Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}"
    )
