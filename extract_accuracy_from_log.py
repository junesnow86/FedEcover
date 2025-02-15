import csv
import re
import sys

if len(sys.argv) != 2:
    print("Usage: python extract_accuracy_from_log.py <method>")
    exit(1)

method = sys.argv[1]
model = "cnn"
dataset = "cifar100"
distribution = "alpha0.5"
capacity = "capacity0"
num_clients = "100clients"

# log_file_path = f"logs/{method}-{model}-{dataset}-{distribution}-{capacity}-{num_clients}.log"
log_file_path = f"logs/20250215/{method}-femnist-cnn.log"

try:
    with open(log_file_path, "r") as file:
        log_content = file.read()
except FileNotFoundError:
    print(f"File {log_file_path} not found.")
    exit(1)

# Use regular expressions to find all occurrences of "Round x" and the corresponding "Aggregated Test Loss" and "Aggregated Test Acc"
pattern = re.compile(
    r"Round (\d+).*?Global Test Loss: ([\d.]+)\s+Global Test Acc: ([\d.]+).*?Round (\d+) use time:",
    re.DOTALL,
)
matches = pattern.findall(log_content)

# Store the extracted data in a list of dictionaries
data = [
    {
        "Round": int(round_num),
        "Aggregated Test Loss": float(test_loss),
        "Aggregated Test Acc": float(test_acc),
    }
    for round_num, test_loss, test_acc, _ in matches
]

# Write the data to a CSV file
# csv_file_path = f"results/{method}_{model}_{dataset}_{distribution}_{capacity}_{num_clients}.csv"
csv_file_path = f"results/20250215/{method}-femnist-cnn.csv"

try:
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = [
            "Round",
            "Aggregated Test Loss",
            "Aggregated Test Acc",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
except IOError:
    print(f"Could not write to {csv_file_path}")
else:
    print(f"Data has been written to {csv_file_path}")
