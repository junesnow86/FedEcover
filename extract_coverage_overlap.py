import csv
import re

date = "1010"

try:
    log_file_path = f"logs/{date}/check_sparsity.log"
    with open(log_file_path, "r") as file:
        log_content = file.read()

    # Step 2: Use regular expressions to find all occurrences of "Round x" and the corresponding "Aggregated Test Loss" and "Aggregated Test Acc"
    pattern = re.compile(
        # r"Round (\d+).*?Global Test Loss: ([\d.]+)\s+Global Test Acc: ([\d.]+).*?Round (\d+) use time:",
        r"Round (\d+).*?\nSparsity: ([\d.]+).*?Coverage: ([\d.]+).*?Round (\d+) use time:",
        re.DOTALL,
    )
    matches = pattern.findall(log_content)

    # Step 3: Store the extracted data in a list of dictionaries
    data = [
        {
            "Round": int(round_num),
            "Coverage": float(coverage),
            "Overlap": float(overlap),
        }
        for round_num, overlap, coverage, _ in matches
    ]

    # Step 4: Write the data to a CSV file
    csv_file_path = f"results/{date}/fedrame_resnet_tiny-imagenet_alpha0.5_coverage_overlap.csv"
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["Round", "Coverage", "Overlap"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

    print(f"Data has been written to {csv_file_path}")
except FileNotFoundError:
    print(f"File {log_file_path} not found.")
    pass