import os
import json
import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "client_splits")

def test_client_splits_exist():
    """Verify all 10 client files and global test dataset exist."""
    assert os.path.exists(DATA_DIR), "client_splits directory missing"
    for i in range(10):
        client_file = os.path.join(DATA_DIR, f"client_{i}.npz")
        assert os.path.exists(client_file), f"Missing {client_file}"
    assert os.path.exists(os.path.join(DATA_DIR, "global_test.npz")), "Missing global_test.npz"
    assert os.path.exists(os.path.join(DATA_DIR, "client_metadata.csv")), "Missing client_metadata.csv"

def test_data_shapes_and_features():
    """Verify 192 features, valid labels, and absence of NaN/Inf."""
    test_data = np.load(os.path.join(DATA_DIR, "global_test.npz"))
    X_test, y_test = test_data["X"], test_data["y"]
    
    assert X_test.shape[1] == 192, f"Expected 192 features, got {X_test.shape[1]}"
    assert len(X_test) == 35069, f"Expected 35069 test samples, got {len(X_test)}"
    assert not np.isnan(X_test).any(), "NaN found in test features"
    assert not np.isinf(X_test).any(), "Inf found in test features"
    assert set(np.unique(y_test)).issubset({0, 1}), "Labels must be binary {0, 1}"

    total_client_samples = 0
    for i in range(10):
        c_data = np.load(os.path.join(DATA_DIR, f"client_{i}.npz"))
        X_c, y_c = c_data["X"], c_data["y"]
        assert X_c.shape[1] == 192
        assert not np.isnan(X_c).any()
        assert not np.isinf(X_c).any()
        assert set(np.unique(y_c)).issubset({0, 1})
        total_client_samples += len(X_c)

    assert total_client_samples == 140272, f"Expected 140272 total train samples across clients, got {total_client_samples}"

def test_model_forward_pass():
    """Verify MLP forward pass output shape and logit properties."""
    from model.mlp_model import MLP
    model = MLP(input_dim=192)
    dummy_input = torch.randn(8, 192)
    output = model(dummy_input)
    assert output.shape == (8, 1), f"Expected output shape (8, 1), got {output.shape}"
