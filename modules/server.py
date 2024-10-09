from typing import List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn

from modules.pruning import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
    extract_submodel_cnn,
    extract_submodel_resnet,
    generate_index_groups,
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
        eta_g: float = 1.0,
        dynamic_eta_g: bool = False,
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

        self.eta_g = eta_g
        self.dynamic_eta_g = dynamic_eta_g

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

        returns = {
            "client_ids": selected_client_ids,
            "client_capacities": selected_client_capacities,
            "submodel_param_indices_dicts": selected_submodel_param_indices_dicts,
            "client_submodels": selected_client_submodels,
        }

        return returns

    def aggregate(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        submodel_param_indices_dicts: List[SubmodelBlockParamIndicesDict],
        client_weights: List[int] = None,
    ):
        self.old_global_params = {
            name: param.clone().detach()
            for name, param in self.global_model.named_parameters()
        }

        if client_weights is None:
            client_weights = [1] * len(selected_client_ids)

        sparsities = []

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

            non_zero_values = averaging_weight_accumulator[averaging_weight_accumulator > 0]
            sparsities.append(non_zero_values.mean().item() / len(selected_client_ids))

        sparsity = sum(sparsities) / len(sparsities)

        # Weighted averaing using old global params and updated global params
        if self.dynamic_eta_g:
            self.eta_g = sparsity

        print(f"Sparsity: {sparsity:.4f}, eta_g: {self.eta_g:.4f}")

        for name, param in self.global_model.named_parameters():
            param.data = (
                (1 - self.eta_g) * self.old_global_params[name]
                + self.eta_g * param.data
            )

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
            layer_out_channels = [64, 128, 256, 512]
            blocks = ["0", "1"]
            convs = ["conv1", "conv2"]
            for _ in range(self.num_clients):
                model_rolling_indices_dict = {
                    "conv1": np.arange(64),
                }
                for i, layer in enumerate(layers):
                    out_channels = layer_out_channels[i]
                    for block in blocks:
                        for conv in convs:
                            model_rolling_indices_dict[f"{layer}.{block}.{conv}"] = (
                                np.arange(out_channels)
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
        client_capacity = self.client_capacities[client_id]

        # Construct a submodel param indices dict for the client
        submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
        if self.model_type == "cnn":
            previous_layer_indices = np.arange(3)
            for layer, indices in model_rolling_indices_dict.items():
                # Note: the layer in the dict is sorted due to Python's dict implementation
                current_layer_indices = indices[: int(client_capacity * len(indices))]
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
                current_layer_indices = indices[: int(client_capacity * len(indices))]
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
                elif conv == "conv2":
                    # There is no downsample layer, the conv2 out indices should stay the same as conv1(of the same block) in indices
                    # due to the residual connection
                    assert np.all(
                        submodel_param_indices_dict[f"{resnet_layer}.{block}.conv1"][
                            "in"
                        ]
                        == submodel_param_indices_dict[f"{resnet_layer}.{block}.conv2"][
                            "out"
                        ]
                    ), "The indices of conv1 and conv2 should be the same due to the residual connection"

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

    def step(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        submodel_param_indices_dicts: List[SubmodelBlockParamIndicesDict],
        client_weights: List[int] = None,
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
            layer_out_channels = [64, 128, 256, 512]
            blocks = ["0", "1"]
            convs = ["conv1", "conv2"]
            for client_id in range(self.num_clients):
                client_capacity = self.client_capacities[client_id]
                model_param_indices_dict = {
                    "conv1": np.arange(int(64 * client_capacity)),
                }
                for i, layer in enumerate(layers):
                    out_channels = layer_out_channels[i]
                    for block in blocks:
                        for conv in convs:
                            model_param_indices_dict[f"{layer}.{block}.{conv}"] = (
                                np.arange(int(out_channels * client_capacity))
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
                elif conv == "conv2":
                    # There is no downsample layer, the conv2 out indices should stay the same as conv1(of the same block) in indices
                    # due to the residual connection
                    assert np.all(
                        submodel_param_indices_dict[f"{resnet_layer}.{block}.conv1"][
                            "in"
                        ]
                        == submodel_param_indices_dict[f"{resnet_layer}.{block}.conv2"][
                            "out"
                        ]
                    ), "The indices of conv1 and conv2 should be the same due to the residual connection"

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

    def step(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        submodel_param_indices_dicts: List[SubmodelBlockParamIndicesDict],
        client_weights: List[int] = None,
    ):
        self.aggregate(
            local_state_dicts,
            selected_client_ids,
            submodel_param_indices_dicts,
            client_weights,
        )


class ServerRD(ServerBase):
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
            global_model,
            num_clients,
            client_capacities,
            model_out_dim,
            model_type,
            select_ratio,
            scaling,
        )

    def get_client_submodel_param_indices_dict(
        self, client_id: int
    ) -> SubmodelBlockParamIndicesDict:
        client_capacity = self.client_capacities[client_id]

        # Construct a submodel param indices dict for the client
        submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
        if self.model_type == "cnn":
            layers = ["layer1.0", "layer2.0", "layer3.0"]
            layer_out_channels = [64, 128, 256]
            previous_layer_indices = np.arange(3)
            for layer, out_channels in zip(layers, layer_out_channels):
                current_layer_indices = np.random.choice(
                    np.arange(out_channels),
                    int(client_capacity * out_channels),
                    replace=False,
                )
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
            current_layer_indices = np.random.choice(
                np.arange(64), int(client_capacity * 64), replace=False
            )
            # Note: Sort the indices
            current_layer_indices = np.sort(current_layer_indices)
            submodel_param_indices_dict["conv1"] = SubmodelLayerParamIndicesDict(
                {"in": previous_layer_indices, "out": current_layer_indices}
            )
            previous_layer_indices = current_layer_indices

            layers = ["layer1", "layer2", "layer3", "layer4"]
            layer_out_channels = [64, 128, 256, 512]
            blocks = ["0", "1"]
            convs = ["conv1", "conv2"]

            for i, layer in enumerate(layers):
                for block in blocks:
                    if layer != "layer1" and block == "0":
                        has_downsample = True
                    else:
                        has_downsample = False

                    for conv in convs:
                        if not has_downsample and conv == "conv2":
                            # There is no downsample layer, the conv2 out indices should stay the same as conv1(of the same block) in indices
                            # due to the residual connection
                            current_layer_indices = submodel_param_indices_dict[
                                f"{layer}.{block}.conv1"
                            ]["in"]
                        else:
                            current_layer_indices = np.random.choice(
                                np.arange(layer_out_channels[i]),
                                int(client_capacity * layer_out_channels[i]),
                                replace=False,
                            )
                            # Note: Sort the indices
                            current_layer_indices = np.sort(current_layer_indices)
                        submodel_param_indices_dict[f"{layer}.{block}.{conv}"] = (
                            SubmodelLayerParamIndicesDict(
                                {
                                    "in": previous_layer_indices,
                                    "out": current_layer_indices,
                                }
                            )
                        )
                        previous_layer_indices = current_layer_indices

                    if has_downsample:
                        # Add the downsample layer
                        submodel_param_indices_dict[f"{layer}.{block}.downsample.0"] = (
                            SubmodelLayerParamIndicesDict(
                                {
                                    "in": submodel_param_indices_dict[
                                        f"{layer}.{block}.conv1"
                                    ]["in"],
                                    "out": submodel_param_indices_dict[
                                        f"{layer}.{block}.conv2"
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

    def step(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        submodel_param_indices_dicts: List[SubmodelBlockParamIndicesDict],
        client_weights: List[int] = None,
    ):
        self.aggregate(
            local_state_dicts,
            selected_client_ids,
            submodel_param_indices_dicts,
            client_weights,
        )


class ServerRDBagging(ServerBase):
    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        client_capacities: List[float],
        model_out_dim: int,
        model_type: str = "cnn",
        select_ratio: float = 0.1,
        scaling: bool = True,
        strategy: str = "p-based-steady",
    ):
        super().__init__(
            global_model,
            num_clients,
            client_capacities,
            model_out_dim,
            model_type,
            select_ratio,
            scaling,
        )

        assert strategy in ["steady", "frequent", "client"]
        self.strategy = strategy
        self.capacity_set = set(client_capacities)
        if strategy == "steady" or strategy == "frequent":
            self.submodel_param_indices_dict_queues = {}  # the key type is str, a p value in string format; the value type is List[SubmodelBlockParamIndicesDict]
            for client_capacity in self.capacity_set:
                self.submodel_param_indices_dict_queues[f"{client_capacity:.2f}"] = []
        else:
            self.submodel_param_indices_dict_queues = [
                [] for _ in range(self.num_clients)
            ]  # the index is the client id; the value type is List[SubmodelBlockParamIndicesDict]

    def get_client_submodel_param_indices_dict(self, client_id: int):
        client_capacity = self.client_capacities[client_id]
        if self.strategy == "steady" or self.strategy == "frequent":
            # Note: `submodel_param_indices_dict_queues` is a quote, so the changes will be reflected in the original object
            submodel_param_indices_dict_queue = self.submodel_param_indices_dict_queues[
                f"{client_capacity:.2f}"
            ]
        else:
            submodel_param_indices_dict_queue = self.submodel_param_indices_dict_queues[
                client_id
            ]

        if len(submodel_param_indices_dict_queue) == 0:
            # Reload the queue
            # Construct a queue of submodel param indices dicts for the client
            if self.model_type == "cnn":
                layers = ["layer1.0", "layer2.0", "layer3.0"]
                layer_out_channels = [64, 128, 256]
                layer_indices_groups = {
                    layer: generate_index_groups(
                        layer_out_channels[i],
                        int(client_capacity * layer_out_channels[i]),
                    )
                    for i, layer in enumerate(layers)
                }

                # Align the lengths of the groups
                queue_length = min(
                    [len(layer_indices_groups[layer]) for layer in layers]
                )
                for layer in layers:
                    layer_indices_groups[layer] = layer_indices_groups[layer][
                        :queue_length
                    ]

                # Note: Sort the indices
                for layer in layers:
                    for i in range(len(layer_indices_groups[layer])):
                        layer_indices_groups[layer][i] = np.sort(
                            layer_indices_groups[layer][i]
                        )

                for i in range(queue_length):
                    submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
                    previous_layer_indices = np.arange(3)
                    for layer in layers:
                        current_layer_indices = layer_indices_groups[layer][i]
                        submodel_param_indices_dict[layer] = (
                            SubmodelLayerParamIndicesDict(
                                {
                                    "in": previous_layer_indices,
                                    "out": current_layer_indices,
                                }
                            )
                        )
                        previous_layer_indices = current_layer_indices

                    # The last fc layer
                    H, W = 4, 4
                    flatten_previous_layer_indices = []
                    for out_channnel_idx in previous_layer_indices:
                        start_idx = out_channnel_idx * H * W
                        end_idx = (out_channnel_idx + 1) * H * W
                        flatten_previous_layer_indices.extend(
                            list(range(start_idx, end_idx))
                        )
                    # Note: Sort the indices
                    flatten_previous_layer_indices = np.sort(
                        flatten_previous_layer_indices
                    )
                    submodel_param_indices_dict["fc"] = SubmodelLayerParamIndicesDict(
                        {
                            "in": flatten_previous_layer_indices,
                            "out": np.arange(self.model_out_dim),
                        }
                    )

                    submodel_param_indices_dict_queue.append(
                        submodel_param_indices_dict
                    )
            elif self.model_type == "resnet":
                layer_indices_groups = {}
                layer_indices_groups["conv1"] = generate_index_groups(
                    64, int(client_capacity * 64)
                )

                layers = ["layer1", "layer2", "layer3", "layer4"]
                layer_out_channels = [64, 128, 256, 512]
                blocks = ["0", "1"]
                convs = ["conv1", "conv2"]

                for i, layer in enumerate(layers):
                    out_channels = layer_out_channels[i]
                    group_size = int(client_capacity * out_channels)
                    for block in blocks:
                        if layer != "layer1" and block == "0":
                            has_downsample = True
                        else:
                            has_downsample = False

                        for conv in convs:
                            if not has_downsample and conv == "conv2":
                                # There is no downsample layer, the conv2 out indices should stay the same as conv1(of the same block) in indices
                                # due to the residual connection
                                continue
                            layer_indices_groups[f"{layer}.{block}.{conv}"] = (
                                generate_index_groups(out_channels, group_size)
                            )

                # Align the lengths of the groups
                queue_length = min(
                    [len(queue) for queue in layer_indices_groups.values()]
                )
                for key in layer_indices_groups.keys():
                    layer_indices_groups[key] = layer_indices_groups[key][:queue_length]

                # Note: Sort the indices
                for key in layer_indices_groups.keys():
                    for i in range(len(layer_indices_groups[key])):
                        layer_indices_groups[key][i] = np.sort(
                            layer_indices_groups[key][i]
                        )

                for i in range(queue_length):
                    submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
                    previous_layer_indices = np.arange(3)
                    current_layer_indices = layer_indices_groups["conv1"][i]
                    submodel_param_indices_dict["conv1"] = (
                        SubmodelLayerParamIndicesDict(
                            {"in": previous_layer_indices, "out": current_layer_indices}
                        )
                    )
                    previous_layer_indices = current_layer_indices

                    for layer in layers:
                        for block in blocks:
                            if layer != "layer1" and block == "0":
                                has_downsample = True
                            else:
                                has_downsample = False

                            for conv in convs:
                                if not has_downsample and conv == "conv2":
                                    # There is no downsample layer, the conv2 out indices should stay the same as conv1(of the same block) in indices
                                    # due to the residual connection
                                    current_layer_indices = submodel_param_indices_dict[
                                        f"{layer}.{block}.conv1"
                                    ]["in"]
                                else:
                                    current_layer_indices = layer_indices_groups[
                                        f"{layer}.{block}.{conv}"
                                    ][i]
                                submodel_param_indices_dict[
                                    f"{layer}.{block}.{conv}"
                                ] = SubmodelLayerParamIndicesDict(
                                    {
                                        "in": previous_layer_indices,
                                        "out": current_layer_indices,
                                    }
                                )
                                previous_layer_indices = current_layer_indices

                            if has_downsample:
                                # Add the downsample layer
                                submodel_param_indices_dict[
                                    f"{layer}.{block}.downsample.0"
                                ] = SubmodelLayerParamIndicesDict(
                                    {
                                        "in": submodel_param_indices_dict[
                                            f"{layer}.{block}.conv1"
                                        ]["in"],
                                        "out": submodel_param_indices_dict[
                                            f"{layer}.{block}.conv2"
                                        ]["out"],
                                    }
                                )

                    H, W = 1, 1
                    flatten_previous_layer_indices = []
                    for out_channnel_idx in previous_layer_indices:
                        start_idx = out_channnel_idx * H * W
                        end_idx = (out_channnel_idx + 1) * H * W
                        flatten_previous_layer_indices.extend(
                            list(range(start_idx, end_idx))
                        )
                    # Note: Sort the indices
                    flatten_previous_layer_indices = np.sort(
                        flatten_previous_layer_indices
                    )
                    submodel_param_indices_dict["fc"] = SubmodelLayerParamIndicesDict(
                        {
                            "in": flatten_previous_layer_indices,
                            "out": np.arange(self.model_out_dim),
                        }
                    )
                    submodel_param_indices_dict_queue.append(
                        submodel_param_indices_dict
                    )
            else:
                raise ValueError("Invalid model type")

        if self.strategy == "steady":
            return submodel_param_indices_dict_queue[-1]
        else:
            return submodel_param_indices_dict_queue.pop()

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

        selected_capacites = [
            f"{self.client_capacities[client_id]:.2f}"
            for client_id in selected_client_ids
        ]
        selected_capacites = list(set(selected_capacites))

        if self.strategy == "steady":
            for client_capacity in selected_capacites:
                self.submodel_param_indices_dict_queues[client_capacity].pop()


class ServerFedRAME(ServerBase):
    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        client_capacities: List[float],
        model_out_dim: int,
        model_type: str = "cnn",
        select_ratio: float = 0.1,
        scaling: bool = True,
        eta_g: float = 1.0,
        dynamic_eta_g: bool = False,
    ):
        super().__init__(
            global_model,
            num_clients,
            client_capacities,
            model_out_dim,
            model_type,
            select_ratio,
            scaling,
            eta_g,
            dynamic_eta_g,
        )

        self.initialize_unused_param_indices_for_layers()

    def initialize_unused_param_indices_for_layers(self):
        self.unused_param_indices_for_layers = {}

        # `pivot_layers` is the layers where the output channels/features can be selected independently
        if self.model_type == "cnn":
            pivot_layers = ["layer1.0", "layer2.0", "layer3.0"]
            out_channel_numbers = [64, 128, 256]
            for i, pivot_layer in enumerate(pivot_layers):
                self.unused_param_indices_for_layers[pivot_layer] = np.arange(
                    out_channel_numbers[i]
                )
        elif self.model_type == "resnet":
            # Pivot layers of ResNet18 include the first conv layer before blocks and the first conv layer of each block
            layers = ["layer1", "layer2", "layer3", "layer4"]
            out_channel_numbers = [64, 128, 256, 512]
            blocks = ["0", "1"]

            self.unused_param_indices_for_layers["conv1"] = np.arange(64)
            for i, layer in enumerate(layers):
                for block in blocks:
                    self.unused_param_indices_for_layers[f"{layer}.{block}.conv1"] = (
                        np.arange(out_channel_numbers[i])
                    )

                    if layer != "layer1" and block == "0":
                        has_downsample = True
                    else:
                        has_downsample = False

                    if has_downsample:
                        self.unused_param_indices_for_layers[
                            f"{layer}.{block}.conv2"
                        ] = np.arange(out_channel_numbers[i])
        else:
            raise ValueError("Invalid model type")

    def random_choose_indices(self, layer: str, out_channels: int, sample_num: int):
        if sample_num > self.unused_param_indices_for_layers[layer].size:
            current_layer_indices = self.unused_param_indices_for_layers[layer].copy()
            remaining_sample_num = sample_num - current_layer_indices.size
            self.unused_param_indices_for_layers[layer] = np.arange(
                out_channels
            )  # Reload
            additional_indices = np.random.choice(
                self.unused_param_indices_for_layers[layer],
                remaining_sample_num,
                replace=False,
            )
            self.unused_param_indices_for_layers[layer] = np.setdiff1d(
                self.unused_param_indices_for_layers[layer], additional_indices
            )
            current_layer_indices = np.sort(
                np.concatenate((current_layer_indices, additional_indices))
            )
        else:
            current_layer_indices = np.sort(
                np.random.choice(
                    self.unused_param_indices_for_layers[layer],
                    sample_num,
                    replace=False,
                )
            )
            self.unused_param_indices_for_layers[layer] = np.setdiff1d(
                self.unused_param_indices_for_layers[layer], current_layer_indices
            )
        return current_layer_indices  # The returned indices are sorted

    def get_client_submodel_param_indices_dict(self, client_id: int):
        client_capacity = self.client_capacities[client_id]

        # Construct a submodel param indices dict for the client
        submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
        if self.model_type == "cnn":
            layers = ["layer1.0", "layer2.0", "layer3.0"]
            out_channel_numbers = [64, 128, 256]
            previous_layer_indices = np.arange(3)
            for layer, out_channels in zip(layers, out_channel_numbers):
                sample_num = int(client_capacity * out_channels)
                current_layer_indices = self.random_choose_indices(
                    layer, out_channels, sample_num
                )
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
            flatten_previous_layer_indices = np.sort(flatten_previous_layer_indices)
            submodel_param_indices_dict["fc"] = SubmodelLayerParamIndicesDict(
                {
                    "in": flatten_previous_layer_indices,
                    "out": np.arange(self.model_out_dim),
                }
            )
        elif self.model_type == "resnet":
            previous_layer_indices = np.arange(3)
            current_layer_indices = self.random_choose_indices(
                "conv1", 64, int(client_capacity * 64)
            )
            submodel_param_indices_dict["conv1"] = SubmodelLayerParamIndicesDict(
                {"in": previous_layer_indices, "out": current_layer_indices}
            )
            previous_layer_indices = current_layer_indices

            layers = ["layer1", "layer2", "layer3", "layer4"]
            layer_out_channels = [64, 128, 256, 512]
            blocks = ["0", "1"]

            for layer_idx, layer in enumerate(layers):
                out_channels = layer_out_channels[layer_idx]
                sample_num = int(client_capacity * out_channels)
                for block in blocks:
                    if layer != "layer1" and block == "0":
                        has_downsample = True
                    else:
                        has_downsample = False

                    for conv in ["conv1", "conv2"]:
                        key = f"{layer}.{block}.{conv}"
                        if not has_downsample and conv == "conv2":
                            # There is no downsample layer, the conv2 out indices should stay the same as conv1(of the same block) in indices
                            # due to the residual connection
                            current_layer_indices = submodel_param_indices_dict[
                                f"{layer}.{block}.conv1"
                            ]["in"]
                        else:
                            current_layer_indices = self.random_choose_indices(
                                key, out_channels, sample_num
                            )

                        submodel_param_indices_dict[key] = (
                            SubmodelLayerParamIndicesDict(
                                {
                                    "in": previous_layer_indices,
                                    "out": current_layer_indices,
                                }
                            )
                        )
                        previous_layer_indices = current_layer_indices

                    if has_downsample:
                        key = f"{layer}.{block}.downsample.0"
                        submodel_param_indices_dict[key] = (
                            SubmodelLayerParamIndicesDict(
                                {
                                    "in": submodel_param_indices_dict[
                                        f"{layer}.{block}.conv1"
                                    ]["in"],
                                    "out": submodel_param_indices_dict[
                                        f"{layer}.{block}.conv2"
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

    def step(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        submodel_param_indices_dicts: List[SubmodelBlockParamIndicesDict],
        client_weights: List[int] = None,
    ):
        self.aggregate(
            local_state_dicts,
            selected_client_ids,
            submodel_param_indices_dicts,
            client_weights,
        )
