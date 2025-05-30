from typing import List, OrderedDict

import numpy as np
import torch.nn as nn

from modules.pruning import (
    SubmodelBlockParamIndicesDict,
    SubmodelLayerParamIndicesDict,
)

from .base import ServerBase


class ServerFD(ServerBase):
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

    def get_client_submodel_param_indices_dict(
        self, client_id: int
    ) -> SubmodelBlockParamIndicesDict:
        client_capacity = self.client_capacities[client_id]

        # Construct a submodel param indices dict for the client
        submodel_param_indices_dict = SubmodelBlockParamIndicesDict()
        if self.model_type == "cnn":
            layers = ["layer1.0", "layer2.0", "layer3.0"]
            layer_out_channels = [64, 128, 256]

            if self.dataset == "femnist":
                previous_layer_indices = np.arange(1)
            else:
                previous_layer_indices = np.arange(3)

            for layer, out_channels in zip(layers, layer_out_channels):
                current_layer_indices = np.sort(
                    np.random.choice(
                        np.arange(out_channels),
                        int(client_capacity * out_channels),
                        replace=False,
                    )
                )
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
            layers = ["layer1.0", "layer2.0"]
            layer_out_channels = [64, 128]
            previous_layer_indices = np.arange(1)

            for layer, out_channels in zip(layers, layer_out_channels):
                current_layer_indices = np.sort(
                    np.random.choice(
                        np.arange(out_channels),
                        int(client_capacity * out_channels),
                        replace=False,
                    )
                )
                submodel_param_indices_dict[layer] = SubmodelLayerParamIndicesDict(
                    {"in": previous_layer_indices, "out": current_layer_indices}
                )
                previous_layer_indices = current_layer_indices

                # Update the neuron selection count
                for idx in current_layer_indices:
                    self.neuron_selection_count[layer][idx] += 1

            # The first fc layer
            H, W = 7, 7
            flatten_previous_layer_indices = []
            for out_channnel_idx in previous_layer_indices:
                start_idx = out_channnel_idx * H * W
                end_idx = (out_channnel_idx + 1) * H * W
                flatten_previous_layer_indices.extend(list(range(start_idx, end_idx)))
            flatten_previous_layer_indices = np.sort(flatten_previous_layer_indices)

            current_layer_indices = np.sort(
                np.random.choice(
                    np.arange(2048), int(client_capacity * 2048), replace=False
                )
            )
            submodel_param_indices_dict["fc1"] = SubmodelLayerParamIndicesDict(
                {
                    "in": flatten_previous_layer_indices,
                    "out": current_layer_indices,
                }
            )
            previous_layer_indices = current_layer_indices
            # Update the neuron selection count
            for idx in current_layer_indices:
                self.neuron_selection_count["fc1"][idx] += 1

            # The second fc layer
            submodel_param_indices_dict["fc2"] = SubmodelLayerParamIndicesDict(
                {
                    "in": previous_layer_indices,
                    "out": np.arange(self.model_out_dim),
                }
            )
        elif self.model_type == "resnet":
            previous_layer_indices = np.arange(3)
            current_layer_indices = np.sort(
                np.random.choice(
                    np.arange(64), int(client_capacity * 64), replace=False
                )
            )
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
                        key = f"{layer}.{block}.{conv}"
                        if not has_downsample and conv == "conv2":
                            # There is no downsample layer, the conv2 out indices should stay the same as conv1(of the same block) in indices
                            # due to the residual connection
                            current_layer_indices = submodel_param_indices_dict[
                                f"{layer}.{block}.conv1"
                            ]["in"]
                        else:
                            current_layer_indices = np.sort(
                                np.random.choice(
                                    np.arange(layer_out_channels[i]),
                                    int(client_capacity * layer_out_channels[i]),
                                    replace=False,
                                )
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
