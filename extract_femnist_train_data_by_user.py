import json
import os

# 假设你的 JSON 文件名为 'data.json'
input_file = "leaf/data/femnist/data/train/all_data_0_niid_0_keep_0_train_9.json"

# 创建一个目录来存储每个用户的 JSON 文件
output_dir = "data/femnist/train"
os.makedirs(output_dir, exist_ok=True)

# 读取 JSON 文件
with open(input_file, "r") as file:
    data = json.load(file)

# 遍历每个用户并生成单独的 JSON 文件
user_count = 0
for user in data["users"]:
    user_data = {"user_id": user, "data": data["user_data"][user]}
    output_file = os.path.join(output_dir, f"{user}.json")
    with open(output_file, "w") as outfile:
        json.dump(user_data, outfile, indent=4)
        user_count += 1

    if user_count % 100 == 0:
        print(f"Processed {user_count} users")

print(
    f"User data files have been created in the '{output_dir}' directory. Total users processed: {user_count}"
)

# 3237 users
