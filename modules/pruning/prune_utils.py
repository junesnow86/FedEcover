from typing import List

import numpy as np


def generate_index_groups(num_elements: int, group_size: int) -> List[np.ndarray]:
    """Packing a given number of indices into groups of a given group size."""
    assert (
        num_elements >= group_size
    ), "Number of elements must be greater than a group size."

    all_indices = np.arange(num_elements)
    np.random.shuffle(all_indices)

    index_groups = []
    total_collected = 0
    while total_collected < num_elements:
        if num_elements - total_collected < group_size:
            num_remaining = num_elements - total_collected
            num_additional_needed = group_size - num_remaining
            group = np.concatenate(
                [all_indices[total_collected:], all_indices[:num_additional_needed]]
            )
            total_collected += num_remaining
        else:
            group = all_indices[total_collected : total_collected + group_size]
            total_collected += group_size

        index_groups.append(group)

    return index_groups
