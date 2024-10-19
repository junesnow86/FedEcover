from typing import List, OrderedDict

import numpy as np
import torch.nn as nn

from modules.pruning import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
)

from .base import ServerBase


class ServerFedRAME(ServerBase):
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
            eta_g=eta_g,
            dynamic_eta_g=dynamic_eta_g,
            norm_type=norm_type,
            param_delta_norm=param_delta_norm,
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
