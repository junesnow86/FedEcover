import json
import os


def extract_femnist_train_data_by_user(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as file:
        data = json.load(file)

    # Iterate over each user and generate an individual JSON file
    # Total 3237 users
    user_count = 0
    for user in data["users"]:
        user_data = {"user_id": user, "data": data["user_data"][user]}
        output_file = os.path.join(output_dir, f"{user}.json")
        with open(output_file, "w") as outfile:
            json.dump(user_data, outfile)
            user_count += 1

        if user_count % 100 == 0:
            print(f"Processed {user_count} users")

    print(
        f"User data files have been created in the '{output_dir}' directory. Total users processed: {user_count}"
    )


def extract_femnist_test_data(input_file, output_file):
    with open(input_file, "r") as file:
        data = json.load(file)

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

    combined_data = {
        "total_samples": total_samples,
        "data": all_user_data,
    }

    with open(output_file, "w") as outfile:
        json.dump(combined_data, outfile)

    print(f"Combined user data file has been created as '{output_file}'.")
    print(f"Total number of samples: {total_samples}")


if __name__ == "__main__":
    input_file = "/home/ljt/research/leaf/data/femnist/data/train/all_data_0_niid_0_keep_0_train_9.json"
    output_dir = "data/femnist/train"
    extract_femnist_train_data_by_user(input_file=input_file, output_dir=output_dir)

    input_file = "/home/ljt/research/leaf/data/femnist/data/test/all_data_0_niid_0_keep_0_test_9.json"
    output_file = "data/femnist/test/test_data.json"
    extract_femnist_test_data(input_file=input_file, output_file=output_file)
