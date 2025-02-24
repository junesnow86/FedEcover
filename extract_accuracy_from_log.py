import csv
import os
import re
import sys

if len(sys.argv) != 2:
    print("Usage: python extract_accuracy_from_log.py <method>")
    exit(1)

method = sys.argv[1]
model = "cnn"
dataset = "cifar100"
distribution = "alpha0.5"
capacity = "capacity2"
num_clients = "10clients"

# sub_dir = "param-sensitivity/Tds100-Tdi10"
# sub_dir = "20250217"
# sub_dir = "iid-gsd"
# sub_dir = "femnist20250219"
sub_dir = "ablation"

log_dir_root = "logs"
if sub_dir:
    log_dir = os.path.join(log_dir_root, sub_dir)
else:
    log_dir = log_dir_root
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

csv_dir_root = "results"
if sub_dir:
    csv_dir = os.path.join(csv_dir_root, sub_dir)
else:
    csv_dir = csv_dir_root
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

log_file_path = os.path.join(
    log_dir,
    f"{method}-{model}-{dataset}-{distribution}-{capacity}-{num_clients}.log",
)
csv_file_path = os.path.join(
    csv_dir,
    f"{method}-{model}-{dataset}-{distribution}-{capacity}-{num_clients}.csv",
)
# log_file_path = os.path.join(log_dir, f"{method}-Tds100-Tdi10.log")
# csv_file_path = os.path.join(csv_dir, f"{method}-Tds100-Tdi10.csv")
# log_file_path = os.path.join(log_dir, f"{method}-Tds200-Tdi10.log")
# csv_file_path = os.path.join(csv_dir, f"{method}-Tds200-Tdi10.csv")
# log_file_path = os.path.join(log_dir, f"{method}-femnist-epochs5-num10.log")
# csv_file_path = os.path.join(csv_dir, f"{method}-femnist-epochs5-num10.csv")
# log_file_path = os.path.join(
#     log_dir, f"{method}-femnist-epochs5-num10-gamma0.9-Tds100-Tdi5.log"
# )
# csv_file_path = os.path.join(
#     csv_dir, f"{method}-femnist-epochs5-num10-gamma0.9-Tds100-Tdi5.csv"
# )
# log_file_path = os.path.join(log_dir, f"{method}-iid-no-gsd.log")
# csv_file_path = os.path.join(csv_dir, f"{method}-iid-no-gsd.csv")

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
    print(f"Error: could not write to {csv_file_path}")
else:
    print(f"Success: data has been written to {csv_file_path}")
