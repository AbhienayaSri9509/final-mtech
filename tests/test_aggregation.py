import pytest
import numpy as np
import torch
from model.mlp_model import MLP
from server.trust_aware_aggregation import (
    compute_aggregation_weights,
    validate_parameter_compatibility,
    aggregate_model_parameters,
    set_torch_model_parameters,
    get_torch_model_parameters,
)

def test_tprs_aggregation_weights():
    # 3 selected clients
    records = [
        {"client_id": 0, "size": 1000, "tprs": 0.8},
        {"client_id": 1, "size": 1000, "tprs": 0.4},
        {"client_id": 2, "size": 2000, "tprs": 0.6},
    ]
    # alpha_i = (tprs * n) / sum(tprs * n)
    # raw: 800, 400, 1200 -> sum = 2400
    # weights: 800/2400=0.3333, 400/2400=0.1667, 1200/2400=0.5
    weights = compute_aggregation_weights(records, method="tprs")
    assert len(weights) == 3
    assert abs(np.sum(weights) - 1.0) < 1e-6
    assert abs(weights[0] - (800 / 2400)) < 1e-6
    assert abs(weights[1] - (400 / 2400)) < 1e-6
    assert abs(weights[2] - (1200 / 2400)) < 1e-6

def test_fedavg_weights():
    records = [
        {"client_id": 0, "size": 1000},
        {"client_id": 1, "size": 3000},
    ]
    weights = compute_aggregation_weights(records, method="fedavg")
    assert abs(weights[0] - 0.25) < 1e-6
    assert abs(weights[1] - 0.75) < 1e-6

def test_aggregation_and_model_update():
    model_a = MLP(input_dim=192)
    model_b = MLP(input_dim=192)

    params_a = get_torch_model_parameters(model_a)
    params_b = get_torch_model_parameters(model_b)

    weights = np.array([0.5, 0.5])
    aggregated = aggregate_model_parameters([params_a, params_b], weights)

    # Set into a new model
    global_model = MLP(input_dim=192)
    set_torch_model_parameters(global_model, aggregated)
    global_params = get_torch_model_parameters(global_model)

    for p_a, p_b, p_g in zip(params_a, params_b, global_params):
        expected = 0.5 * p_a + 0.5 * p_b
        np.testing.assert_allclose(p_g, expected, atol=1e-5)
