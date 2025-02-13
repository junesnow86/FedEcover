import os
import time

import numpy as np


def calculate_model_size(model, print_result=True, unit="MB"):
    total_params = 0
    for param in model.parameters():
        # Multiply the size of each dimension to get the total number of elements
        total_params += param.numel()

    # Assuming each parameter is a 32-bit float
    memory_bytes = total_params * 4
    memory_kilobytes = memory_bytes / 1024
    memory_megabytes = memory_kilobytes / 1024

    if print_result:
        print(f"Total parameters: {total_params}")
        print(
            f"Memory Usage: {memory_bytes} bytes ({memory_kilobytes:.2f} KB / {memory_megabytes:.2f} MB)"
        )

    if unit == "KB":
        return memory_kilobytes
    elif unit == "MB":
        return memory_megabytes
    else:
        return memory_bytes


def measure_time(repeats: int = 1):
    """Decorator, measure the function time costs with mean and variance.

    Args:
        repeats (int, optional): Repeat times for measuring. Defaults to 10.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            timings = []
            for _ in range(repeats):
                start_time = time.time()
                func_res = func(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                timings.append(elapsed)
            np_times = np.array(timings)
            average_time = np.mean(np_times)
            variance = np.var(np_times)
            # logging.info(f"[{func.__name__}] Average time over {repeats} runs: {average_time:.6f} seconds")
            print(
                f"[{func.__name__}] Average time over {repeats} runs: {average_time:.6f} seconds"
            )
            if repeats > 1:
                # logging.info(f"[{func.__name__}] Variance of times: {variance:.6f}")
                print(f"[{func.__name__}] Variance of times: {variance:.6f}")
            return func_res

        return wrapper

    return decorator


def get_user_id_by_idx(idx, base_dir):
    user_ids = sorted(os.listdir(base_dir))

    if idx < 0 or idx >= len(user_ids):
        raise IndexError("Index out of range for user IDs.")

    return user_ids[idx]
