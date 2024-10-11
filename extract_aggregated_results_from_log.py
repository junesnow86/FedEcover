import re
import csv



methods = ["fedavg"]
models = ["cnn"]
datasets = ["cifar10", "cifar100"]
distributions = ["iid", "alpha0.1", "alpha0.5"]

date = "1009"

for method in methods:
    for model in models:
        for dataset in datasets:
            for distribution in distributions:
                try:
                    log_file_path = f"logs/{date}/{method}_{model}_{dataset}_{distribution}.log"
                    # log_file_path = f"logs/{date}/{method}_{model}_{dataset}_{distribution}_epochs10.log"
                    with open(log_file_path, "r") as file:
                        log_content = file.read()

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

                    # Step 4: Write the data to a CSV file
                    csv_file_path = f"results/{date}/{method}_{model}_{dataset}_{distribution}_aggregated_test_results.csv"
                    with open(csv_file_path, "w", newline="") as csvfile:
                        fieldnames = ["Round", "Aggregated Test Loss", "Aggregated Test Acc"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        writer.writeheader()
                        writer.writerows(data)

                    print(f"Data has been written to {csv_file_path}")
                except FileNotFoundError:
                    print(f"File {log_file_path} not found.")
                    pass
