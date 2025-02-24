#!/bin/bash

# Define the methods to iterate over
methods=("fedecover" "fedavg" "fd" "fedrolex" "heterofl")
# methods=("fedecover-no-gsd" "fedavg-no-gsd" "fd-no-gsd" "fedrolex-no-gsd" "heterofl-no-gsd")
# methods=("fedecover-gamma1.0" "fedavg-gamma1.0" "fd-gamma1.0" "fedrolex-gamma1.0" "heterofl-gamma1.0")
# methods=("fedecover-gamma0.9" "fedecover-gamma0.8" "fedecover-gamma0.7" "fedecover-gamma0.6" "fedecover-gamma0.5")
# methods=("fedecover-gamma0.95" "fedecover-gamma0.9" "fedecover-gamma0.85" "fedecover-gamma0.8")
# methods=("fd-gamma0.95" "fd-gamma0.9" "fd-gamma0.85" "fd-gamma0.8")
# methods=("fedrolex-gamma0.95" "fedrolex-gamma0.9" "fedrolex-gamma0.85" "fedrolex-gamma0.8")
# methods=("heterofl-gamma0.95" "heterofl-gamma0.9" "heterofl-gamma0.85" "heterofl-gamma0.8")
# methods=("fedavg-gamma0.95" "fedavg-gamma0.9" "fedavg-gamma0.85" "fedavg-gamma0.8")

# Loop through each method and execute the Python script
for method in "${methods[@]}"; do
    python3 extract_accuracy_from_log.py "$method"
done