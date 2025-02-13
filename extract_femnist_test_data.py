import json

# 假设你的 JSON 文件名为 'data.json'
input_file = "leaf/data/femnist/data/test/all_data_0_niid_0_keep_0_test_9.json"
output_file = "data/femnist/test/test_data.json"

# 读取 JSON 文件
with open(input_file, "r") as file:
    data = json.load(file)

# 整合所有用户的数据
all_user_data = {
    "x": [],
    "y": [],
}
total_samples = 0

for user in data["users"]:
    user_x = data["user_data"][user]["x"]
    user_y = data["user_data"][user]["y"]
    all_user_data["x"].extend(user_x)
    all_user_data["y"].extend(user_y)
    assert len(user_x) == len(user_y)
    total_samples += len(user_x)


total_data_count = len(all_user_data["x"])

# 创建一个新的 JSON 对象
combined_data = {
    "total_samples": total_samples,
    "data": all_user_data,
}

# 将整合后的数据写入新的 JSON 文件
with open(output_file, "w") as outfile:
    json.dump(combined_data, outfile)

print(f"Combined user data file has been created as '{output_file}'.")
print(f"Total number of samples: {total_samples}")
