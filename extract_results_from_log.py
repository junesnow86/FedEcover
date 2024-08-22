import csv
import os
import re

data_dir = "results_0821"

with open("results_0821/rd_base_lr_decay_unbalanced.log", "r") as file:
    lines = file.readlines()

round_pattern = re.compile(r"Round (\d+)")
subset_pattern = re.compile(
    r"Subset (\d+)\s+Train Loss: ([\d.]+)\s+Test Loss: ([\d.]+)\s+Test Acc: ([\d.]+)"
)
aggregated_pattern = re.compile(
    r"Aggregated Test Loss: ([\d.]+)\s+Aggregated Test Acc: ([\d.]+)"
)

round_number = 0
train_loss_dict = {}
test_loss_dict = {}
test_acc_dict = {}

for line in lines:
    round_match = round_pattern.search(line)
    subset_match = subset_pattern.search(line)
    aggregated_match = aggregated_pattern.search(line)
    if round_match:
        round_number = int(round_match.group(1))
    elif subset_match:
        subset_number = int(subset_match.group(1))
        train_loss = float(subset_match.group(2))
        test_loss = float(subset_match.group(3))
        test_acc = float(subset_match.group(4))

        if round_number not in train_loss_dict:
            train_loss_dict[round_number] = {}
            test_loss_dict[round_number] = {}
            test_acc_dict[round_number] = {}

        train_loss_dict[round_number][subset_number] = train_loss
        test_loss_dict[round_number][subset_number] = test_loss
        test_acc_dict[round_number][subset_number] = test_acc
    elif aggregated_match:
        aggregated_test_loss = float(aggregated_match.group(1))
        aggregated_test_acc = float(aggregated_match.group(2))

        test_loss_dict[round_number]["Aggregated"] = aggregated_test_loss
        test_acc_dict[round_number]["Aggregated"] = aggregated_test_acc

# Prepare data for CSV
train_loss_data = []
test_loss_data = []
test_acc_data = []

for round_number in sorted(train_loss_dict.keys()):
    train_loss_row = {"Round": round_number}
    test_loss_row = {"Round": round_number}
    test_acc_row = {"Round": round_number}

    for i in range(1, 11):
        train_loss_row[f"Subset {i}"] = train_loss_dict[round_number].get(i, "")
        test_loss_row[f"Subset {i}"] = test_loss_dict[round_number].get(i, "")
        test_acc_row[f"Subset {i}"] = test_acc_dict[round_number].get(i, "")

    test_loss_row["Aggregated"] = test_loss_dict[round_number].get("Aggregated", "")
    test_acc_row["Aggregated"] = test_acc_dict[round_number].get("Aggregated", "")

    train_loss_data.append(train_loss_row)
    test_loss_data.append(test_loss_row)
    test_acc_data.append(test_acc_row)

# Write train loss to CSV
with open(os.path.join(data_dir, "rd_base_lr_decay_unbalanced_train_loss.csv"), "w", newline="") as csvfile:
    fieldnames = ["Round"] + [f"Subset {i}" for i in range(1, 11)]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(train_loss_data)

# Write test loss to CSV
with open(os.path.join(data_dir, "rd_base_lr_decay_unbalanced_test_loss.csv"), "w", newline="") as csvfile:
    fieldnames = ["Round"] + [f"Subset {i}" for i in range(1, 11)] + ["Aggregated"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(test_loss_data)

# Write test acc to CSV
with open(os.path.join(data_dir, "rd_base_lr_decay_unbalanced_test_acc.csv"), "w", newline="") as csvfile:
    fieldnames = ["Round"] + [f"Subset {i}" for i in range(1, 11)] + ["Aggregated"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(test_acc_data)
