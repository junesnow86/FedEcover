import re
import csv

# 打开日志文件
with open('results_0820/rd_base.log', 'r') as file:
    lines = file.readlines()

# 初始化存储数据的列表
data = []
round_number = 0

# 正则表达式匹配aggregated class acc行
pattern = re.compile(r'Aggregated Class Acc: \{(.+?)\}')
# pattern = re.compile(r'Round (\d+), Aggregated Test Acc: [\d.]+	Class Acc: \{(.*?)\}', re.DOTALL)
# pattern = re.compile(r'Round (\d+), Whole Aggregated Test Acc: [\d.]+	Class Acc: \{(.*?)\}', re.DOTALL)

# 逐行读取日志文件
for line in lines:
    match = pattern.search(line)
    if match:
        round_number += 1
        class_acc = match.group(1)
        # class_acc = match.group(2)
        # 将字符串转换为字典
        class_acc_dict = eval(f"{{{class_acc}}}")
        # 将数据存储在列表中
        data.append([round_number] + [class_acc_dict[i] for i in range(10)])

# 写入CSV文件
with open('results_0820/aggregated_class_acc_rd_base.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(['Round'] + [f'Class_{i}_Acc' for i in range(10)])
    # 写入数据
    writer.writerows(data)
