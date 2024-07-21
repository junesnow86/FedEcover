import csv
import re


def extract_acc_class_acc(log_file_path):
    with open(log_file_path, "r") as file:
        log_content = file.read()

    # Pattern to match each round's details including subsets and aggregated results
    # round_pattern = re.compile(
    #     r"Round (\d+), Subset (\d+), Test Acc: ([0-9.]+)\tClass Acc: ({.*?})"
    #     r"|Round (\d+), Aggregated Test Acc: ([0-9.]+)\tClass Acc: ({.*?})"
    # )
    round_pattern = re.compile(
        r"Round (\d+), Subset (\d+), (?:Group Averaging )?Test Acc: ([0-9.]+)\tClass Acc: ({.*?})"
        r"|Round (\d+), Aggregated Test Acc: ([0-9.]+)\tClass Acc: ({.*?})"
    )

    results = {}
    for match in round_pattern.finditer(log_content):
        if match.group(1):  # This is a subset match
            round_number = int(match.group(1))
            subset_number = int(match.group(2))
            test_acc = float(match.group(3))
            class_acc = eval(
                match.group(4)
            )  # Convert string representation of dict to dict

            if round_number not in results:
                results[round_number] = {}
            results[round_number][f"Subset {subset_number}"] = {
                "Test Acc": test_acc,
                "Class Acc": class_acc,
            }
        else:  # This is an aggregated match
            round_number = int(match.group(5))
            aggregated_test_acc = float(match.group(6))
            aggregated_class_acc = eval(
                match.group(7)
            )  # Convert string representation of dict to dict

            if round_number not in results:
                results[round_number] = {}
            results[round_number]["Aggregated"] = {
                "Test Acc": aggregated_test_acc,
                "Class Acc": aggregated_class_acc,
            }

    return results


def write_overall_results_to_csv(results, csv_file_path):
    # 首先确定最大的子集数量以设置列标题
    max_subsets = max(
        len(round_data) - 1 for round_data in results.values()
    )  # 减1是为了排除聚合结果
    fieldnames = (
        ["Round"] + [f"Subset {i+1}" for i in range(max_subsets)] + ["Aggregated"]
    )

    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for round_number, round_data in sorted(results.items()):
            row = {"Round": round_number}
            for subset_name, subset_data in round_data.items():
                if "Subset" in subset_name:
                    row[subset_name] = subset_data["Test Acc"]
                else:  # Aggregated data
                    row["Aggregated"] = subset_data["Test Acc"]
            writer.writerow(row)


def write_class_wise_results_to_csv(results, csv_file_path):
    # 确定所有可能的类别
    all_classes = set()
    for round_data in results.values():
        for data in round_data.values():
            all_classes.update(data["Class Acc"].keys())
    all_classes = sorted(all_classes)  # 排序以保持一致性

    # 构建列标题
    fieldnames = ["Round", "Subset"] + all_classes

    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for round_number, round_data in sorted(results.items()):
            for subset_name, subset_data in round_data.items():
                # 处理subset名称，aggregated也视为一个subset
                subset_number = (
                    subset_name.split(" ")[1]
                    if "Subset" in subset_name
                    else "Aggregated"
                )
                row = {"Round": round_number, "Subset": subset_number}
                for class_name in all_classes:
                    row[class_name] = subset_data["Class Acc"].get(class_name, "")
                writer.writerow(row)


def write_aggregated_class_wise_results_to_csv(results, csv_file_path):
    # 初始化一个集合来收集所有类别的名称
    all_classes = set()
    for round_data in results.values():
        aggregated_data = round_data.get("Aggregated", {})
        class_acc = aggregated_data.get("Class Acc", {})
        all_classes.update(class_acc.keys())
    all_classes = sorted(all_classes)  # 对类别名称进行排序

    # 构建CSV文件的列标题
    fieldnames = ["Round"] + ["Class " + str(class_name) for class_name in all_classes]

    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历每一轮的结果
        for round_number, round_data in sorted(results.items()):
            if "Aggregated" in round_data:
                # 准备这一轮的数据行
                row = {"Round": round_number}
                class_acc = round_data["Aggregated"].get("Class Acc", {})
                for class_name in all_classes:
                    row["Class " + str(class_name)] = class_acc.get(class_name, "")
                writer.writerow(row)


# 假设`results`已经从之前的步骤中生成
# csv_file_path = "statistics/0721/position_class_wise_acc_results.csv"
csv_file_path = "statistics/0721/position_aggregated_class_wise_acc_results.csv"

# Example usage
log_file_path = "results/position_heterofl_unbalanced_classes.log"
# log_file_path = "position_heterofl_unbalanced_classes_100rounds_0720.log"
results = extract_acc_class_acc(log_file_path)
# write_overall_results_to_csv(results, csv_file_path)
# write_class_wise_results_to_csv(results, csv_file_path)
write_aggregated_class_wise_results_to_csv(results, csv_file_path)
print(results)
