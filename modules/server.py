from typing import List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn

from modules.pruning import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
    extract_submodel_cnn,
    extract_submodel_resnet,
)


class ServerBase:
    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        client_capacities: List[float],
        model_out_dim: int,
        model_type: str = "cnn",
        select_ratio: float = 0.1,
        scaling: bool = True,
    ):
        assert len(client_capacities) == num_clients
        assert model_type in ["cnn", "resnet"]

        self.global_model = global_model
        self.model_type = model_type
        self.num_clients = num_clients
        self.client_capacities = client_capacities
        self.model_out_dim = model_out_dim
        self.select_ratio = select_ratio
        self.scaling = scaling

    def get_client_submodel_param_indices_dict(self, client_id: int):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def distribute(
        self,
    ) -> Tuple[
        List[int], List[float], List[SubmodelBlockParamIndicesDict], List[nn.Module]
    ]:
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
        if self.model_type == "cnn":
            selected_client_submodels = [
                extract_submodel_cnn(
                    original_model=self.global_model,
                    submodel_param_indices_dict=selected_submodel_param_indices_dicts[
                        i
                    ],
                    p=1 - selected_client_capacities[i],
                    scaling=True,
                )
                for i in range(len(selected_client_ids))
            ]
        elif self.model_type == "resnet":
            selected_client_submodels = [
                extract_submodel_resnet(
                    original_model=self.global_model,
                    submodel_param_indices_dict=selected_submodel_param_indices_dicts[
                        i
                    ],
                    p=1 - selected_client_capacities[i],
                    scaling=True,
                )
                for i in range(len(selected_client_ids))
            ]
        else:
            raise ValueError("Invalid model type")

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

    def step(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )


class ServerFedRolex(ServerBase):
    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        client_capacities: List[float],
        model_out_dim: int,
        model_type: str = "cnn",
        select_ratio: float = 0.1,
        scaling: bool = True,
        rolling_step: int = -1,
    ):
        super().__init__(
            global_model=global_model,
            num_clients=num_clients,
            client_capacities=client_capacities,
            model_out_dim=model_out_dim,
            model_type=model_type,
            select_ratio=select_ratio,
            scaling=scaling,
        )

        self.rolling_step = rolling_step
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
        elif self.model_type == "resnet":
            layers = ["layer1", "layer2", "layer3", "layer4"]
            out_channels = [64, 128, 256, 512]
            blocks = ["0", "1"]
            convs = ["conv1", "conv2"]
            for _ in range(self.num_clients):
                model_rolling_indices_dict = {
                    "conv1": np.arange(64),
                }
                for i, layer in enumerate(layers):
                    out_channel = out_channels[i]
                    for block in blocks:
                        for conv in convs:
                            model_rolling_indices_dict[f"{layer}.{block}.{conv}"] = (
                                np.arange(out_channel)
                            )
                self.model_rolling_indices_dicts.append(model_rolling_indices_dict)
        else:
            raise ValueError("Invalid model type")

    def roll_indices(self):
        for model_rolling_indices_dict in self.model_rolling_indices_dicts:
            for layer, indices in model_rolling_indices_dict.items():
                model_rolling_indices_dict[layer] = np.roll(
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
            for layer, indices in model_rolling_indices_dict.items():
                # Note: the layer in the dict is sorted due to Python's dict implementation
                current_layer_indices = indices[
                    : int(self.client_capacities[client_id] * len(indices))
                ]
                # Note: Sort the indices
                current_layer_indices = np.sort(current_layer_indices)
                submodel_param_indices_dict[layer] = SubmodelLayerParamIndicesDict(
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
        elif self.model_type == "resnet":
            previous_layer_indices = np.arange(3)
            for layer, indices in model_rolling_indices_dict.items():
                # Note: the layer in the dict is sorted due to Python's dict implementation
                current_layer_indices = indices[
                    : int(self.client_capacities[client_id] * len(indices))
                ]
                # Note: Sort the indices
                current_layer_indices = np.sort(current_layer_indices)
                submodel_param_indices_dict[layer] = SubmodelLayerParamIndicesDict(
                    {"in": previous_layer_indices, "out": current_layer_indices}
                )
                previous_layer_indices = current_layer_indices

                if len(layer.split(".")) == 1:
                    continue

                resnet_layer = layer.split(".")[0]
                block = layer.split(".")[1]
                conv = layer.split(".")[2]
                if resnet_layer != "layer1" and block == "0" and conv == "conv2":
                    # Add the downsample layer
                    downsample_layer = f"{resnet_layer}.{block}.downsample.0"
                    submodel_param_indices_dict[downsample_layer] = (
                        SubmodelLayerParamIndicesDict(
                            {
                                "in": submodel_param_indices_dict[
                                    f"{resnet_layer}.{block}.conv1"
                                ]["in"],
                                "out": submodel_param_indices_dict[
                                    f"{resnet_layer}.{block}.conv2"
                                ]["out"],
                            }
                        )
                    )

            # The last fc layer
            H, W = 1, 1
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
        else:
            raise ValueError("Invalid model type")

        return submodel_param_indices_dict

    def distribute(self):
        return super().distribute()

    def aggregate(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        submodel_param_indices_dicts: List[SubmodelBlockParamIndicesDict],
        client_weights: List[int],
    ):
        super().aggregate(
            local_state_dicts,
            selected_client_ids,
            submodel_param_indices_dicts,
            client_weights,
        )

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


class ServerHeteroFL(ServerBase):
    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        client_capacities: List[float],
        model_out_dim: int,
        model_type: str = "cnn",
        select_ratio: float = 0.1,
        scaling: bool = True,
    ):
        super().__init__(
            global_model=global_model,
            num_clients=num_clients,
            client_capacities=client_capacities,
            model_out_dim=model_out_dim,
            model_type=model_type,
            select_ratio=select_ratio,
            scaling=scaling,
        )

        self.initialize_param_indices_dicts()

    def initialize_param_indices_dicts(self):
        self.model_param_indices_dicts = []
        if self.model_type == "cnn":
            for client_id in range(self.num_clients):
                client_capacity = self.client_capacities[client_id]
                model_param_indices_dict = {
                    "layer1.0": np.arange(int(64 * client_capacity)),
                    "layer2.0": np.arange(int(128 * client_capacity)),
                    "layer3.0": np.arange(int(256 * client_capacity)),
                }
                self.model_param_indices_dicts.append(model_param_indices_dict)
        elif self.model_type == "resnet":
            layers = ["layer1", "layer2", "layer3", "layer4"]
            out_channels = [64, 128, 256, 512]
            blocks = ["0", "1"]
            convs = ["conv1", "conv2"]
            for client_id in range(self.num_clients):
                client_capacity = self.client_capacities[client_id]
                model_param_indices_dict = {
                    "conv1": np.arange(int(64 * client_capacity)),
                }
                for i, layer in enumerate(layers):
                    out_channel = out_channels[i]
                    for block in blocks:
                        for conv in convs:
                            model_param_indices_dict[f"{layer}.{block}.{conv}"] = (
                                np.arange(int(out_channel * client_capacity))
                            )
                self.model_param_indices_dicts.append(model_param_indices_dict)
        else:
            raise ValueError("Invalid model type")

    def get_client_submodel_param_indices_dict(
        self, client_id: int
    ) -> SubmodelBlockParamIndicesDict:
        model_param_indices_dict = self.model_param_indices_dicts[client_id]

        # Construct a submodel param indices dict for the client
        submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
        if self.model_type == "cnn":
            previous_layer_indices = np.arange(3)
            for layer, indices in model_param_indices_dict.items():
                # Note: the layer in the dict is sorted due to Python's dict implementation
                submodel_param_indices_dict[layer] = SubmodelLayerParamIndicesDict(
                    {"in": previous_layer_indices, "out": indices}
                )
                previous_layer_indices = indices

            # The last fc layer
            H, W = 4, 4
            flatten_previous_layer_indices = []
            for out_channnel_idx in previous_layer_indices:
                start_idx = out_channnel_idx * H * W
                end_idx = (out_channnel_idx + 1) * H * W
                flatten_previous_layer_indices.extend(list(range(start_idx, end_idx)))
            flatten_previous_layer_indices = np.sort(flatten_previous_layer_indices)
            submodel_param_indices_dict["fc"] = SubmodelLayerParamIndicesDict(
                {
                    "in": flatten_previous_layer_indices,
                    "out": np.arange(self.model_out_dim),
                }
            )
        elif self.model_type == "resnet":
            previous_layer_indices = np.arange(3)
            for layer, indices in model_param_indices_dict.items():
                # Note: the layer in the dict is sorted due to Python's dict implementation
                submodel_param_indices_dict[layer] = SubmodelLayerParamIndicesDict(
                    {"in": previous_layer_indices, "out": indices}
                )
                previous_layer_indices = indices

                if len(layer.split(".")) == 1:
                    continue

                resnet_layer = layer.split(".")[0]
                block = layer.split(".")[1]
                conv = layer.split(".")[2]
                if resnet_layer != "layer1" and block == "0" and conv == "conv2":
                    # Add the downsample layer
                    downsample_layer = f"{resnet_layer}.{block}.downsample.0"
                    submodel_param_indices_dict[downsample_layer] = (
                        SubmodelLayerParamIndicesDict(
                            {
                                "in": submodel_param_indices_dict[
                                    f"{resnet_layer}.{block}.conv1"
                                ]["in"],
                                "out": submodel_param_indices_dict[
                                    f"{resnet_layer}.{block}.conv2"
                                ]["out"],
                            }
                        )
                    )

            # The last fc layer
            H, W = 1, 1
            flatten_previous_layer_indices = []
            for out_channnel_idx in previous_layer_indices:
                start_idx = out_channnel_idx * H * W
                end_idx = (out_channnel_idx + 1) * H * W
                flatten_previous_layer_indices.extend(list(range(start_idx, end_idx)))
            flatten_previous_layer_indices = np.sort(flatten_previous_layer_indices)
            submodel_param_indices_dict["fc"] = SubmodelLayerParamIndicesDict(
                {
                    "in": flatten_previous_layer_indices,
                    "out": np.arange(self.model_out_dim),
                }
            )
        else:
            raise ValueError("Invalid model type")

        return submodel_param_indices_dict

    def distribute(self):
        return super().distribute()

    def aggregate(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        submodel_param_indices_dicts: List[SubmodelBlockParamIndicesDict],
        client_weights: List[int],
    ):
        super().aggregate(
            local_state_dicts,
            selected_client_ids,
            submodel_param_indices_dicts,
            client_weights,
        )

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
