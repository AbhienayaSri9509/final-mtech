import os
import json
import pytest
from server.experiment_runner import run_experiment

def test_full_experiment_pipeline_run():
    """Verify execution of full experiment pipeline, saving metrics.json and model checkpoint."""
    cfg = {
        "name": "test_pipeline_unit",
        "algorithm": "TrustFed-6G",
        "rounds": 2,
        "num_clients": 10,
        "clients_per_round": 4,
        "local_epochs": 1,
        "batch_size": 256,
        "learning_rate": 0.001,
        "seed": 42,
        "max_batches": 4,  # Fast execution for automated unit testing
        "enable_privacy": True,
        "enable_dynamic_trust": True,
    }
    result = run_experiment(cfg)
    assert result is not None
    assert "final_metrics" in result
    assert "round_history" in result
    assert len(result["round_history"]) == 2

    # Check that disk output exists
    exp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "experiments", "test_pipeline_unit")
    assert os.path.exists(os.path.join(exp_dir, "metrics.json"))
    assert os.path.exists(os.path.join(exp_dir, "model_checkpoint.pth"))

    # Verify JSON content
    with open(os.path.join(exp_dir, "metrics.json"), "r") as f:
        data = json.load(f)
        assert data["experiment_id"] == "test_pipeline_unit"
        assert 0.0 <= data["final_metrics"]["accuracy"] <= 1.0
