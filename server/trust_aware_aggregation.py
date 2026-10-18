"""
Trust-Aware Model Aggregation Module for TrustFed-6G.

Formula:
    alpha_i = (TPRS_i * n_i) / sum_{j in S} (TPRS_j * n_j)

Where:
    - S: set of selected clients in the current round
    - TPRS_i: composite score of client i
    - n_i: number of local samples belonging to client i
    - alpha_i: normalized aggregation weight for client i (sum(alpha_i) == 1.0)
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch import nn


def compute_aggregation_weights(
    client_records: List[Dict[str, Any]],
    method: str = "tprs",
) -> np.ndarray:
    """
    Compute normalized aggregation weights alpha_i over the selected clients.
    
    Supports:
    - 'tprs': alpha_i = (TPRS_i * n_i) / sum(TPRS_j * n_j)
    - 'fedavg': alpha_i = n_i / sum(n_j)
    - 'uniform': alpha_i = 1 / K
    """
    if not client_records:
        raise ValueError("Cannot compute aggregation weights for an empty client list.")

    sizes = np.array([max(1, int(c.get("size", c.get("num_samples", 1)))) for c in client_records], dtype=np.float64)

    if method == "uniform":
        return np.ones(len(client_records), dtype=np.float64) / len(client_records)

    elif method == "fedavg":
        total_size = np.sum(sizes)
        if total_size <= 0:
            return np.ones(len(client_records), dtype=np.float64) / len(client_records)
        return sizes / total_size

    elif method == "tprs":
        tprs_scores = np.array([max(0.001, float(c.get("tprs", 0.5))) for c in client_records], dtype=np.float64)
        raw_weights = tprs_scores * sizes
        total_weight = np.sum(raw_weights)

        # Fallback if zero or invalid
        if total_weight <= 0 or np.isnan(total_weight) or np.isinf(total_weight):
            return sizes / np.sum(sizes)

        return raw_weights / total_weight

    else:
        raise ValueError(f"Unknown aggregation method: '{method}'. Supported: 'tprs', 'fedavg', 'uniform'.")


def validate_parameter_compatibility(
    client_parameters_list: List[List[np.ndarray]],
) -> None:
    """
    Ensure all client model updates have identical layer count,
    layer shapes, and datatypes.
    """
    if not client_parameters_list:
        raise ValueError("No client parameters provided for aggregation.")

    ref_shapes = [p.shape for p in client_parameters_list[0]]
    ref_dtypes = [p.dtype for p in client_parameters_list[0]]

    for idx, params in enumerate(client_parameters_list[1:], start=1):
        if len(params) != len(ref_shapes):
            raise ValueError(
                f"Client {idx} parameter length ({len(params)}) does not match reference ({len(ref_shapes)})."
            )
        for layer_idx, (p, ref_s, ref_d) in enumerate(zip(params, ref_shapes, ref_dtypes)):
            if p.shape != ref_s:
                raise ValueError(
                    f"Client {idx} layer {layer_idx} shape {p.shape} does not match expected {ref_s}."
                )
            if np.isnan(p).any() or np.isinf(p).any():
                raise ValueError(
                    f"Client {idx} layer {layer_idx} contains NaN or Infinite weights."
                )


def aggregate_model_parameters(
    client_parameters_list: List[List[np.ndarray]],
    weights: np.ndarray,
) -> List[np.ndarray]:
    """
    Perform weighted linear combination of model parameters:
        theta_global = sum_i (alpha_i * theta_i)
    """
    validate_parameter_compatibility(client_parameters_list)

    if len(weights) != len(client_parameters_list):
        raise ValueError(
            f"Weights count ({len(weights)}) does not match client updates count ({len(client_parameters_list)})."
        )

    # Ensure weights are normalized to 1.0
    w_norm = weights / np.sum(weights)

    num_layers = len(client_parameters_list[0])
    aggregated_params = []

    for layer_idx in range(num_layers):
        layer_comb = np.zeros_like(client_parameters_list[0][layer_idx], dtype=np.float64)
        for client_idx, client_params in enumerate(client_parameters_list):
            layer_comb += w_norm[client_idx] * client_params[layer_idx].astype(np.float64)
        aggregated_params.append(layer_comb.astype(client_parameters_list[0][layer_idx].dtype))

    return aggregated_params


def set_torch_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load parameter arrays into a PyTorch model state_dict safely."""
    state_dict = model.state_dict()
    if len(state_dict) != len(parameters):
        raise ValueError(
            f"Model state_dict entries ({len(state_dict)}) != parameters provided ({len(parameters)})."
        )

    new_state = {}
    for (k, v), param in zip(state_dict.items(), parameters):
        new_state[k] = torch.as_tensor(param, dtype=v.dtype, device=v.device)

    model.load_state_dict(new_state)


def get_torch_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as numpy copies."""
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]
