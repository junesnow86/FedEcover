from typing import List, OrderedDict

import numpy as np
import torch.nn as nn

from modules.pruning import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
)

from .base import ServerBase


class ServerStatic(ServerBase):
    def __init__(
        self,
        global_model: nn.Module,
        dataset: str,
        num_clients: int,
        client_capacities: List[float],
        model_out_dim: int,
        model_type: str = "cnn",
        select_ratio: float = 0.1,
        scaling: bool = True,
        norm_type: str = "ln",
        eta_g: float = 1.0,
        dynamic_eta_g: bool = False,
        param_delta_norm: str = "mean",
    ):
        super().__init__(
            global_model=global_model,
            dataset=dataset,
            num_clients=num_clients,
            client_capacities=client_capacities,
            model_out_dim=model_out_dim,
            model_type=model_type,
            select_ratio=select_ratio,
            scaling=scaling,
            norm_type=norm_type,
            eta_g=eta_g,
            dynamic_eta_g=dynamic_eta_g,
            param_delta_norm=param_delta_norm,
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
