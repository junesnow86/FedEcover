#!/bin/bash

# Define the methods to iterate over
methods=("fedecover" "fedavg" "fd" "fedrolex" "heterofl")

# Loop through each method and execute the Python script
for method in "${methods[@]}"; do
    python3 extract_accuracy_from_log.py "$method"
done