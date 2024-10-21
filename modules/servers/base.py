from typing import List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn

from modules.pruning import (
    SubmodelBlockParamIndicesDict,
    extract_submodel_cnn,
    extract_submodel_resnet,
)


class ServerBase:
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
        norm_type: str = "sbn",
        eta_g: float = 1.0,
        dynamic_eta_g: bool = False,
        param_delta_norm: str = "mean",
        global_lr_decay: bool = False,
        gamma: float = 0.5,
        decay_steps: List[int] = [50, 100],
    ):
        assert len(client_capacities) == num_clients
        assert model_type in ["cnn", "resnet"]
        assert param_delta_norm in ["mean", "uniform"]

        self.global_model = global_model
        self.dataset = dataset
        self.model_type = model_type
        self.num_clients = num_clients
        self.client_capacities = client_capacities
        self.model_out_dim = model_out_dim
        self.select_ratio = select_ratio
        self.scaling = scaling

        self.eta_g = eta_g
        self.dynamic_eta_g = dynamic_eta_g
        self.norm_type = norm_type

        self.round = 0
        self.global_lr_decay = global_lr_decay
        self.gamma = gamma
        self.decay_steps = decay_steps
        self.param_delta_norm = param_delta_norm

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
                    dataset=self.dataset,
                    norm_type=self.norm_type,
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
        if client_weights is None:
            client_weights = [1] * len(selected_client_ids)

        if self.param_delta_norm == "uniform":
            # client_weight_base = sum(client_weights)
            mean_capacity = sum(self.client_capacities) / len(self.client_capacities)
            client_weight_base = sum(client_weights) * mean_capacity

        self.old_global_params = {
            name: param.clone().detach()
            for name, param in self.global_model.named_parameters()
        }

        named_parameter_overlaps = {}
        named_parameter_coverages = {}
        named_parameter_coverages_ablation = {}
        num_full_models = 0
        for client_id in selected_client_ids:
            if self.client_capacities[client_id] >= 1.0:
                num_full_models += 1

        named_parameters_delta = {}

        for param_name, param in self.global_model.named_parameters():
            param_accumulator = torch.zeros_like(param.data)
            param_delta_accumulator = torch.zeros_like(param.data)

            if "bias" in param_name:
                averaging_weight_accumulator = torch.zeros(param.data.size(0))
            elif "weight" in param_name:
                averaging_weight_accumulator = torch.zeros(
                    param.data.size(0), param.data.size(1)
                )
            else:
                raise ValueError(f"Invalid parameter name: {param_name}")

            # Accumulate param value and weight for averaging
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
                    param_delta_accumulator[np.ix_(out_indices, in_indices)] += (
                        local_param[
                            np.ix_(range(len(out_indices)), range(len(in_indices)))
                        ]
                        - self.old_global_params[param_name][
                            np.ix_(out_indices, in_indices)
                        ]
                    ) * client_weight
                    averaging_weight_accumulator[np.ix_(out_indices, in_indices)] += (
                        client_weight
                    )
                elif "bias" in param_name:
                    param_accumulator[out_indices] += (
                        local_param[range(len(out_indices))] * client_weight
                    )
                    param_delta_accumulator[out_indices] += (
                        local_param[range(len(out_indices))]
                        - self.old_global_params[param_name][out_indices]
                    ) * client_weight
                    averaging_weight_accumulator[out_indices] += client_weight

            # Normalize the accumulated param value
            nonzero_indices = averaging_weight_accumulator > 0
            if param.dim() == 4:
                if self.param_delta_norm == "mean":
                    param_delta_accumulator[nonzero_indices] /= (
                        averaging_weight_accumulator[nonzero_indices][:, None, None]
                    )
                elif self.param_delta_norm == "uniform":
                    print(f"Client weight base: {client_weight_base}")
                    # param_delta_accumulator[nonzero_indices] /= sum(client_weights)
                    param_delta_accumulator[nonzero_indices] /= client_weight_base
                else:
                    raise ValueError(
                        f"Invalid delta normalization: {self.param_delta_norm}"
                    )
            elif param.dim() == 2 or param.dim() == 1:
                if self.param_delta_norm == "mean":
                    param_delta_accumulator[nonzero_indices] /= (
                        averaging_weight_accumulator[nonzero_indices]
                    )
                elif self.param_delta_norm == "uniform":
                    print(f"Client weight base: {client_weight_base}")
                    # param_delta_accumulator[nonzero_indices] /= sum(client_weights)
                    param_delta_accumulator[nonzero_indices] /= client_weight_base
                else:
                    raise ValueError(
                        f"Invalid delta normalization: {self.param_delta_norm}"
                    )
            else:
                raise ValueError(f"Invalid parameter dimension: {param.dim()}")

            named_parameters_delta[param_name] = param_delta_accumulator

            non_zero_values = averaging_weight_accumulator[
                averaging_weight_accumulator > 0
            ]
            overlap = non_zero_values.mean().item() / len(selected_client_ids)
            named_parameter_overlaps[param_name] = overlap

            coverage = (
                averaging_weight_accumulator > 0
            ).sum().item() / averaging_weight_accumulator.numel()
            named_parameter_coverages[param_name] = coverage

            coverage_ablation = (
                (averaging_weight_accumulator - num_full_models) > 0
            ).sum().item() / averaging_weight_accumulator.numel()
            named_parameter_coverages_ablation[param_name] = coverage_ablation

        mean_overlap = sum(named_parameter_overlaps.values()) / len(
            named_parameter_overlaps
        )
        mean_coverage = sum(named_parameter_coverages.values()) / len(
            named_parameter_coverages
        )
        mean_coverage_ablation = sum(named_parameter_coverages_ablation.values()) / len(
            named_parameter_coverages_ablation
        )

        if self.dynamic_eta_g:
            self.eta_g = mean_overlap

        print(
            f"eta_g: {self.eta_g:.4f}, mean Overlap: {mean_overlap:.4f}, mean Coverage: {mean_coverage:.4f}, mean Coverage Ablation: {mean_coverage_ablation:.4f}"
        )

        # Update global model
        for param_name, param in self.global_model.named_parameters():
            param.data += self.eta_g * named_parameters_delta[param_name]
            # param.data = (1 - self.eta_g) * self.old_global_params[param_name] + self.eta_g * param.data
            print(
                f"Param name: {param_name}, param overlap: {named_parameter_overlaps[param_name]:.4f}, param coverage: {named_parameter_coverages[param_name]:.4f}, param coverage ablation: {named_parameter_coverages_ablation[param_name]:.4f}"
            )

        self.round += 1
        if self.global_lr_decay and self.round in self.decay_steps:
            self.eta_g *= self.gamma
            print(
                f"Decaying global learning rate at round {self.round}, new eta_g: {self.eta_g:.4f}"
            )

    def step(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )
