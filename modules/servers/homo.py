import copy
from typing import List, OrderedDict

import numpy as np
import torch.nn as nn

from modules.aggregation import federated_averaging
from modules.pruning import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
    extract_submodel_cnn,
    extract_submodel_resnet,
)
from .base import ServerBase


class ServerHomo(ServerBase):
    def __init__(
        self,
        global_model: nn.Module,
        dataset: str,
        num_clients: int,
        client_capacities: List[float],
        model_out_dim: int,
        model_type: str = "cnn",
        select_ratio: float = 0.1,
        norm_type: str = "sbn",
        eta_g: float = 1.0,
        dynamic_eta_g: bool = False,
        global_lr_decay: bool = False,
        gamma: float = 0.5,
        decay_steps: List[int] = [50, 100],
    ):
        super().__init__(
            global_model=global_model,
            dataset=dataset,
            num_clients=num_clients,
            client_capacities=client_capacities,
            model_out_dim=model_out_dim,
            model_type=model_type,
            select_ratio=select_ratio,
            norm_type=norm_type,
            eta_g=eta_g,
            dynamic_eta_g=dynamic_eta_g,
            global_lr_decay=global_lr_decay,
            gamma=gamma,
            decay_steps=decay_steps,
        )

        self.client_capacity = min(client_capacities)
        self.global_model = self.initialize_global_model(global_model)

    def initialize_global_model(self, original_model: nn.Module):
        if self.model_type == "cnn":
            self.hidden_layer_neuron_indices_dict = {
                "layer1.0": np.arange(int(64 * self.client_capacity)),
                "layer2.0": np.arange(int(128 * self.client_capacity)),
                "layer3.0": np.arange(int(256 * self.client_capacity)),
            }

            # Construct the submodel param indices dict
            self.submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
            previous_layer_indices = np.arange(3)
            for layer, indices in self.hidden_layer_neuron_indices_dict.items():
                # Note: the layer in the dict is sorted due to Python's dict implementation
                self.submodel_param_indices_dict[layer] = SubmodelLayerParamIndicesDict(
                    {"in": previous_layer_indices, "out": indices}
                )
                previous_layer_indices = indices

            # The last fc layer
            # H, W = 1, 1
            H, W = 4, 4
            flatten_previous_layer_indices = []
            for out_channnel_idx in previous_layer_indices:
                start_idx = out_channnel_idx * H * W
                end_idx = (out_channnel_idx + 1) * H * W
                flatten_previous_layer_indices.extend(list(range(start_idx, end_idx)))
            flatten_previous_layer_indices = np.sort(flatten_previous_layer_indices)
            self.submodel_param_indices_dict["fc"] = SubmodelLayerParamIndicesDict(
                {
                    "in": flatten_previous_layer_indices,
                    "out": np.arange(self.model_out_dim),
                }
            )

            # Extract submodel
            return extract_submodel_cnn(
                original_model=original_model,
                submodel_param_indices_dict=self.submodel_param_indices_dict,
                p=1 - self.client_capacity,
                scaling=True,
            )
        elif self.model_type == "resnet":
            self.hidden_layer_neuron_indices_dict = {
                "conv1": np.arange(int(64 * self.client_capacity)),
            }
            layers = ["layer1", "layer2", "layer3", "layer4"]
            layer_out_channels = [64, 128, 256, 512]
            blocks = ["0", "1"]
            convs = ["conv1", "conv2"]
            for i, layer in enumerate(layers):
                out_channels = layer_out_channels[i]
                for block in blocks:
                    for conv in convs:
                        self.hidden_layer_neuron_indices_dict[
                            f"{layer}.{block}.{conv}"
                        ] = np.arange(int(out_channels * self.client_capacity))

            # Construct the submodel param indices dict
            self.submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
            previous_layer_indices = np.arange(3)
            for layer, indices in self.hidden_layer_neuron_indices_dict.items():
                # Note: the layer in the dict is sorted due to Python's dict implementation
                self.submodel_param_indices_dict[layer] = SubmodelLayerParamIndicesDict(
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
                    self.submodel_param_indices_dict[downsample_layer] = (
                        SubmodelLayerParamIndicesDict(
                            {
                                "in": self.submodel_param_indices_dict[
                                    f"{resnet_layer}.{block}.conv1"
                                ]["in"],
                                "out": self.submodel_param_indices_dict[
                                    f"{resnet_layer}.{block}.conv2"
                                ]["out"],
                            }
                        )
                    )
                elif conv == "conv2":
                    # There is no downsample layer, the conv2 out indices should stay the same as conv1(of the same block) in indices
                    # due to the residual connection
                    assert np.all(
                        self.submodel_param_indices_dict[
                            f"{resnet_layer}.{block}.conv1"
                        ]["in"]
                        == self.submodel_param_indices_dict[
                            f"{resnet_layer}.{block}.conv2"
                        ]["out"]
                    ), "The indices of conv1 and conv2 should be the same due to the residual connection"

            # The last fc layer
            H, W = 1, 1
            flatten_previous_layer_indices = []
            for out_channnel_idx in previous_layer_indices:
                start_idx = out_channnel_idx * H * W
                end_idx = (out_channnel_idx + 1) * H * W
                flatten_previous_layer_indices.extend(list(range(start_idx, end_idx)))
            flatten_previous_layer_indices = np.sort(flatten_previous_layer_indices)
            self.submodel_param_indices_dict["fc"] = SubmodelLayerParamIndicesDict(
                {
                    "in": flatten_previous_layer_indices,
                    "out": np.arange(self.model_out_dim),
                }
            )

            # Extract submodel
            return extract_submodel_resnet(
                original_model=original_model,
                submodel_param_indices_dict=self.submodel_param_indices_dict,
                p=1 - self.client_capacity,
                scaling=True,
                dataset=self.dataset,
                norm_type=self.norm_type,
            )
        else:
            raise ValueError("Invalid model type")

    def distribute(self):
        selected_client_ids = np.random.choice(
            self.num_clients, int(self.num_clients * self.select_ratio), replace=False
        )

        selected_client_submodels = [
            copy.deepcopy(self.global_model) for _ in range(len(selected_client_ids))
        ]

        returns = {
            "client_ids": selected_client_ids,
            "client_submodels": selected_client_submodels,
        }

        return returns

    def aggregate(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        client_weights: List[int] = None,
    ):
        if client_weights is None:
            client_weights = [1] * len(selected_client_ids)
        aggregated_state_dict = federated_averaging(local_state_dicts, client_weights)
        self.global_model.load_state_dict(aggregated_state_dict)

    def step(
        self,
        local_state_dicts: List[OrderedDict],
        selected_client_ids: List[int],
        client_weights: List[int] = None,
    ):
        self.aggregate(local_state_dicts, selected_client_ids, client_weights)
