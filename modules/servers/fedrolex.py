from typing import List, OrderedDict

import numpy as np
import torch.nn as nn

from modules.pruning import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
)

from .base import ServerBase


class ServerFedRolex(ServerBase):
    def __init__(
        self,
        global_model: nn.Module,
        dataset: str,
        num_clients: int,
        client_capacities: List[float],
        model_out_dim: int,
        model_type: str = "cnn",
        num_selected_clients: int = 10,
        scaling: bool = True,
        norm_type: str = "sbn",
        eta_g: float = 1.0,
        global_lr_decay: bool = False,
        gamma: float = 0.5,
        decay_steps: List[int] = [50, 100],
        rolling_step: int = -1,
    ):
        super().__init__(
            global_model=global_model,
            dataset=dataset,
            num_clients=num_clients,
            client_capacities=client_capacities,
            model_out_dim=model_out_dim,
            model_type=model_type,
            num_selected_clients=num_selected_clients,
            scaling=scaling,
            norm_type=norm_type,
            eta_g=eta_g,
            global_lr_decay=global_lr_decay,
            gamma=gamma,
            decay_steps=decay_steps,
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
        elif self.model_type == "femnistcnn":
            for _ in range(self.num_clients):
                model_rolling_indices_dict = {
                    "layer1.0": np.arange(64),
                    "layer2.0": np.arange(128),
                    "fc1": np.arange(2048),
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

    def get_client_submodel_param_indices_dict(self, client_id: int):
        model_rolling_indices_dict = self.model_rolling_indices_dicts[client_id]
        client_capacity = self.client_capacities[client_id]

        # Construct a submodel param indices dict for the client
        submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
        if self.model_type == "cnn":
            if self.dataset == "femnist":
                previous_layer_indices = np.arange(1)
            else:
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

                # Update the neuron selection count
                for idx in current_layer_indices:
                    self.neuron_selection_count[layer][idx] += 1

            # The last fc layer
            if self.dataset in ["cifar10", "cifar100", "femnist"]:
                H, W = 4, 4
            elif self.dataset == "celeba":
                H, W = 16, 16
            else:
                raise ValueError("Invalid dataset")

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
        elif self.model_type == "femnistcnn":
            previous_layer_indices = np.arange(1)

            for layer, indices in model_rolling_indices_dict.items():
                # Note: the layer in the dict is sorted due to Python's dict implementation
                current_layer_indices = indices[: int(client_capacity * len(indices))]
                # Note: Sort the indices
                current_layer_indices = np.sort(current_layer_indices)

                if layer == "fc1":
                    H, W = 7, 7
                    flatten_previous_layer_indices = []
                    for out_channnel_idx in previous_layer_indices:
                        start_idx = out_channnel_idx * H * W
                        end_idx = (out_channnel_idx + 1) * H * W
                        flatten_previous_layer_indices.extend(
                            list(range(start_idx, end_idx))
                        )
                    flatten_previous_layer_indices = np.sort(
                        flatten_previous_layer_indices
                    )
                    previous_layer_indices = flatten_previous_layer_indices

                submodel_param_indices_dict[layer] = SubmodelLayerParamIndicesDict(
                    {"in": previous_layer_indices, "out": current_layer_indices}
                )
                previous_layer_indices = current_layer_indices

                # Update the neuron selection count
                for idx in current_layer_indices:
                    self.neuron_selection_count[layer][idx] += 1

            # The last fc layer
            submodel_param_indices_dict["fc2"] = SubmodelLayerParamIndicesDict(
                {
                    "in": previous_layer_indices,
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
                    ), (
                        "The indices of conv1 and conv2 should be the same due to the residual connection"
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
        self.roll_indices()
