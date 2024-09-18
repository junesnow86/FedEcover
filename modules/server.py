from typing import List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn

from modules.pruning import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
    extract_submodel_cnn,
)

# 每个用户持有一个dropout rate `p`，表示该用户支持的模型容量
# 每个用户持有一个当前轮索引记录表，记录当前轮中该用户使用的子模型的每一层的参数索引
# 每个用户持有一个索引队列，记录后续轮中该用户使用的子模型的每一层的参数索引

# 每个用户、每一层，都有一个输出通道/特征索引列表/张量，每一轮进行rolling，取前`capacity`比例的索引对应的参数来构建本轮分发给该用户的子模型


class ServerFedRolex:
    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        client_capacities: List[float],
        model_out_dim: int,
        model_type: str = "cnn",
        select_ratio: float = 0.1,
        rolling_step: int = 1,
        scaling: bool = True,
    ):
        assert len(client_capacities) == num_clients
        assert model_type in ["cnn", "resnet"]

        self.model_type = model_type

        self.global_model = global_model
        self.num_clients = num_clients
        self.client_capacities = client_capacities
        self.model_out_dim = model_out_dim
        self.select_ratio = select_ratio
        self.rolling_step = rolling_step
        self.scaling = scaling

        self.round = 0

        self.initialize_rolling_indices_dicts()

    def initialize_rolling_indices_dicts(self):
        self.model_rolling_indices_dicts = []
        if self.model_type == "cnn":
            for _ in range(self.num_clients):
                model_rolling_indices_dict = {
                    "layer1.0": np.arange(64),
                    "layer2.0": np.arange(128),
                    "layer3.0": np.arange(256),
                }
                self.model_rolling_indices_dicts.append(model_rolling_indices_dict)
        elif self.model_type == "resnet18":
            pass
        else:
            raise ValueError("Invalid model type")

    def roll_indices(self):
        for model_rolling_indices_dict in self.model_rolling_indices_dicts:
            for layer_name, indices in model_rolling_indices_dict.items():
                model_rolling_indices_dict[layer_name] = np.roll(
                    indices,
                    shift=self.rolling_step,
                )

    def get_client_submodel_param_indices_dict(
        self, client_id: int
    ) -> SubmodelBlockParamIndicesDict:
        model_rolling_indices_dict = self.model_rolling_indices_dicts[client_id]

        # Construct a submodel param indices dict for the client
        submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
        if self.model_type == "cnn":
            previous_layer_indices = np.arange(3)
            for layer_name, indices in model_rolling_indices_dict.items():
                current_layer_indices = indices[
                    : int(self.client_capacities[client_id] * len(indices))
                ]
                # Note: Sort the indices
                current_layer_indices = np.sort(current_layer_indices)
                submodel_param_indices_dict[layer_name] = SubmodelLayerParamIndicesDict(
                    {"in": previous_layer_indices, "out": current_layer_indices}
                )
                previous_layer_indices = current_layer_indices

            # The last fc layer
            H, W = 4, 4
            flatten_previous_layer_indices = []
            for out_channnel_idx in previous_layer_indices:
                start_idx = out_channnel_idx * H * W
                end_idx = (out_channnel_idx + 1) * H * W
                flatten_previous_layer_indices.extend(list(range(start_idx, end_idx)))
            # Note: Sort the indices
            flatten_previous_layer_indices = np.sort(flatten_previous_layer_indices)
            submodel_param_indices_dict["fc"] = SubmodelLayerParamIndicesDict(
                {
                    "in": flatten_previous_layer_indices,
                    "out": np.arange(self.model_out_dim),
                }
            )
        elif self.model_type == "resnet18":
            pass

        return submodel_param_indices_dict

    def distribute(self) -> Tuple[List[int], List[float], List[nn.Module]]:
        """Generate submodels for each client, according to each client's capacity and its rolling indices dict"""
        selected_client_ids = np.random.choice(
            self.num_clients, int(self.num_clients * self.select_ratio), replace=False
        )
        selected_client_capacities = [
            self.client_capacities[client_id] for client_id in selected_client_ids
        ]
        selected_submodel_param_indices_dicts = [
            self.get_client_submodel_param_indices_dict(client_id)
            for client_id in selected_client_ids
        ]
        selected_client_submodels = [
            extract_submodel_cnn(
                original_model=self.global_model,
                submodel_param_indices_dict=selected_submodel_param_indices_dicts[i],
                p=1 - selected_client_capacities[i],
                scaling=True,
            )
            for i in range(len(selected_client_ids))
        ]

        return (
            selected_client_ids,
            selected_client_capacities,
            selected_submodel_param_indices_dicts,
            selected_client_submodels,
        )

    def aggregate(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        submodel_param_indices_dicts: List[SubmodelBlockParamIndicesDict],
        client_weights: List[int],
    ):
        for param_name, param in self.global_model.named_parameters():
            param_accumulator = torch.zeros_like(param.data)

            if "bias" in param_name:
                averaging_weight_accumulator = torch.zeros(param.data.size(0))
            elif "weight" in param_name:
                averaging_weight_accumulator = torch.zeros(
                    param.data.size(0), param.data.size(1)
                )
            else:
                raise ValueError(f"Invalid parameter name: {param_name}")

            for local_state_dict, submodel_param_indices_dict, client_weight in zip(
                local_state_dicts, submodel_param_indices_dicts, client_weights
            ):
                if self.scaling:
                    if "weight" in param_name:
                        try:
                            local_param = local_state_dict[
                                param_name.replace(".weight", ".0.weight")
                            ]
                        except KeyError:
                            local_param = local_state_dict[param_name]
                    elif "bias" in param_name:
                        try:
                            local_param = local_state_dict[
                                param_name.replace(".bias", ".0.bias")
                            ]
                        except KeyError:
                            local_param = local_state_dict[param_name]
                    else:
                        raise ValueError(f"Invalid parameter name: {param_name}")
                else:
                    local_param = local_state_dict[param_name]

                if "bias" in param_name:
                    key = param_name.replace(".bias", "")
                elif "weight" in param_name:
                    key = param_name.replace(".weight", "")
                else:
                    raise ValueError(f"Invalid parameter name: {param_name}")

                submodel_param_indices = submodel_param_indices_dict[key]
                in_indices, out_indices = (
                    submodel_param_indices["in"],
                    submodel_param_indices["out"],
                )

                # Check in_indices and out_indices are sorted
                assert np.all(in_indices == np.sort(in_indices))
                assert np.all(out_indices == np.sort(out_indices))

                if "weight" in param_name:
                    param_accumulator[np.ix_(out_indices, in_indices)] += (
                        local_param[
                            np.ix_(range(len(out_indices)), range(len(in_indices)))
                        ]
                        * client_weight
                    )
                    averaging_weight_accumulator[np.ix_(out_indices, in_indices)] += (
                        client_weight
                    )
                elif "bias" in param_name:
                    param_accumulator[out_indices] += (
                        local_param[range(len(out_indices))] * client_weight
                    )
                    averaging_weight_accumulator[out_indices] += client_weight

            nonzero_indices = averaging_weight_accumulator > 0
            if param.dim() == 4:
                # This is a convolution weight
                param.data[nonzero_indices] = (
                    param_accumulator[nonzero_indices]
                    / averaging_weight_accumulator[nonzero_indices][:, None, None]
                )
            elif param.dim() == 2:
                # This is a linear weight
                param.data[nonzero_indices] = (
                    param_accumulator[nonzero_indices]
                    / averaging_weight_accumulator[nonzero_indices]
                )
            elif param.dim() == 1:
                # This is a bias
                param.data[nonzero_indices] = (
                    param_accumulator[nonzero_indices]
                    / averaging_weight_accumulator[nonzero_indices]
                )
            else:
                raise ValueError(f"Invalid parameter dimension: {param.dim()}")

    def step(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        submodel_param_indices_dicts: List[SubmodelBlockParamIndicesDict],
        client_weights: List[int],
    ):
        self.aggregate(
            local_state_dicts,
            selected_client_ids,
            submodel_param_indices_dicts,
            client_weights,
        )
        self.roll_indices()
        self.round += 1
