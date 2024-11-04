import csv
import os
import re

date = "1031"

# methods = ["fedavg", "heterofl", "fedrolex", "fedrd", "fedrame"]
# methods = ["fedavg", "heterofl", "fedrolex", "fedrd", "fedrame", "fedrame2"]
# methods = ["fedavg", "heterofl", "fedrolex", "fedrame"]
# methods = ["fedrd", "fedrame"]
# methods = ["fedrd", "fedrd-noaug-debugging"]
# methods = ["fedrd", "fedrd-aug"]
# methods = ["fedrame", "fedrame-aug"]
# methods = ["fedavg", "fedrd", "fedrame"]
# methods = ["fedrd", "fedrd-20epochs"]
# methods = ["fedrame", "fedrame-20epochs"]
# methods = ["fedrame", "fedrame2"]
# methods = ["fedrd", "fedrd-20epochs", "fedrame", "fedrame-20epochs"]
methods = ["fedavg", "fedavg-no-gsd"]
# methods = ["heterofl", "heterofl-no-gsd"]
# methods = ["fedrolex", "fedrolex-no-gsd"]
# methods = ["fedrd", "fedrd-no-gsd"]
# methods = ["fedrame", "fedrame-no-gsd"]

models = ["cnn"]
# models = ["resnet"]

# datasets = ["cifar10"]
datasets = ["cifar100"]
# datasets = ["cifar10", "cifar100"]
# datasets = ["tinyimagenet"]

# distributions = ["iid", "alpha0.5", "alpha0.1"]
# distributions = ["alpha0.1"]
distributions = ["alpha0.5"]
# distributions = ["alpha1.0"]

capacity = "capacity0"
# capacity = "capacity2"

# num_clients = 10
num_clients = 100


for method in methods:
    for model in models:
        for dataset in datasets:
            for distribution in distributions:
                if num_clients is not None:
                    log_file_path = f"logs/{date}/{method}-{model}-{dataset}-{distribution}-{capacity}-{num_clients}clients.log"
                else:
                    log_file_path = f"logs/{date}/{method}-{model}-{dataset}-{distribution}-{capacity}.log"
                    # log_file_path = f"logs/{date}/{method}-{model}-{dataset}-{distribution}.log"
                    # log_file_path = f"logs/{date}/{method}_{model}_{dataset}_{distribution}.log"
                    # log_file_path = f"logs/{date}/{method}_{model}_{dataset}_{distribution}_epochs10.log"

                try:
                    with open(log_file_path, "r") as file:
                        log_content = file.read()
                except FileNotFoundError:
                    print(f"File {log_file_path} not found.")

                # Step 2: Use regular expressions to find all occurrences of "Round x" and the corresponding "Aggregated Test Loss" and "Aggregated Test Acc"
                pattern = re.compile(
                    # r"Round (\d+).*?Aggregated Test Loss: ([\d.]+)\s+Aggregated Test Acc: ([\d.]+).*?Round (\d+) use time:",
                    r"Round (\d+).*?Global Test Loss: ([\d.]+)\s+Global Test Acc: ([\d.]+).*?Round (\d+) use time:",
                    re.DOTALL,
                )
                matches = pattern.findall(log_content)

                # Step 3: Store the extracted data in a list of dictionaries
                data = [
                    {
                        "Round": int(round_num),
                        "Aggregated Test Loss": float(test_loss),
                        "Aggregated Test Acc": float(test_acc),
                    }
                    for round_num, test_loss, test_acc, _ in matches
                ]

                results_dir = f"results/{date}"
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                # Step 4: Write the data to a CSV file
                if num_clients is not None:
                    csv_file_path = f"results/{date}/{method}_{model}_{dataset}_{distribution}_{capacity}_{num_clients}clients.csv"
                else:
                    csv_file_path = f"results/{date}/{method}_{model}_{dataset}_{distribution}_{capacity}.csv"

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
