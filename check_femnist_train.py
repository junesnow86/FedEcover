import json

# 假设你的 JSON 文件名为 'data.json'
with open('leaf/data/femnist/data/train/all_data_0_niid_0_keep_0_train_9.json', 'r') as file:
    data = json.load(file)

# 获取第一个用户的名字
first_user = data['users'][0]

# 打印第一个用户的 user_data
print(data['user_data'][first_user])
