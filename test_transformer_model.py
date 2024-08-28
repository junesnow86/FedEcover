import torch
import torch.nn as nn
import torch.optim as optim
from modules.models import Transformer


src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 3
d_ff = 2048
max_seq_length = 100
dropout = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = Transformer(
    src_vocab_size,
    tgt_vocab_size,
    num_heads,
    num_layers,
    d_model,
    d_ff,
    max_seq_length,
    dropout,
)
print("model created")
print(transformer)

transformer.to(device)

print(
    f"[Create Model] Memory Allocated after model creation: {torch.cuda.memory_allocated(device) / 1024 ** 2} MB"
)
print(
    f"[Create Model] Memory Reserved after model creation: {torch.cuda.memory_reserved(device) / 1024 ** 2} MB"
)

# Generate random sample data
src_data = torch.randint(
    1, src_vocab_size, (64, max_seq_length)
)  # (batch_size, seq_length)
tgt_data = torch.randint(
    1, tgt_vocab_size, (64, max_seq_length)
)  # (batch_size, seq_length)
print("data created")

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()


for epoch in range(1):
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(
        output.contiguous().view(-1, tgt_vocab_size),
        tgt_data[:, 1:].contiguous().view(-1),
    )
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

transformer.eval()

# Generate random sample validation data
val_src_data = torch.randint(
    1, src_vocab_size, (64, max_seq_length)
)  # (batch_size, seq_length)
val_tgt_data = torch.randint(
    1, tgt_vocab_size, (64, max_seq_length)
)  # (batch_size, seq_length)

with torch.no_grad():
    val_src_data = val_src_data.to(device)
    val_tgt_data = val_tgt_data.to(device)
    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(
        val_output.contiguous().view(-1, tgt_vocab_size),
        val_tgt_data[:, 1:].contiguous().view(-1),
    )
    print(f"Validation Loss: {val_loss.item()}")
