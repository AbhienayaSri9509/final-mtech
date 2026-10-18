"""
Reproducible Federated Learning Experiment Runner for TrustFed-6G.

Orchestrates:
    - Experiment A: Benign FedAvg
    - Experiment B: Attacked FedAvg (Malicious clients {0, 3, 7}, sign-flip)
    - Experiment C: TrustFed-6G (TPRS Selection, Trust-Aware Aggregation, Dynamic Trust Defense)
    - Privacy Experiments (Adaptive Noise)
    - Ablation Studies (A1 - A6)
"""

import os
import json
import time
import random
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model.mlp_model import MLP
from trust.tprs import (
    calculate_quality_score,
    calculate_resource_score,
    calculate_network_score,
    calculate_privacy_suitability,
    calculate_tprs,
    rank_and_select_clients,
    update_dynamic_trust,
)
from server.trust_aware_aggregation import (
    compute_aggregation_weights,
    aggregate_model_parameters,
    set_torch_model_parameters,
    get_torch_model_parameters,
)
from privacy.adaptive_dp import add_adaptive_noise
from security.poisoning import poison_parameters, compute_update_cosine_similarity


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_DIR = os.path.join(PROJECT_ROOT, "data", "client_splits")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def load_client_dataset(client_id: int) -> TensorDataset:
    path = os.path.join(CLIENT_DIR, f"client_{client_id}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Client dataset {path} does not exist.")
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32).reshape(-1, 1)
    return TensorDataset(X, y)


def load_global_test_dataset() -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(CLIENT_DIR, "global_test.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global test dataset {path} does not exist.")
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.int64)


def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    preds = (probs >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def local_train(
    client_id: int,
    global_parameters: List[np.ndarray],
    epochs: int = 1,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> Tuple[List[np.ndarray], int, float]:
    """Train client locally on its partition and return (params, sample_count, loss)."""
    dataset = load_client_dataset(client_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MLP(input_dim=192)
    set_torch_model_parameters(model, global_parameters)
    model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    steps = 0
    for _ in range(epochs):
        for b_idx, (xb, yb) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    avg_loss = float(total_loss / max(1, steps))
    params = get_torch_model_parameters(model)
    return params, len(dataset), avg_loss


def run_experiment(
    config: Dict[str, Any],
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run an end-to-end reproducible federated learning experiment.
    """
    exp_name = config.get("name", "experiment_" + str(int(time.time())))
    algorithm = config.get("algorithm", "TrustFed-6G")
    rounds = int(config.get("rounds", 5))
    num_clients = int(config.get("num_clients", 10))
    clients_per_round = int(config.get("clients_per_round", 7))
    epochs = int(config.get("local_epochs", 1))
    batch_size = int(config.get("batch_size", 256))
    lr = float(config.get("learning_rate", 0.001))
    seed = int(config.get("seed", 42))
    max_batches = config.get("max_batches", None)

    # Attack & Privacy configs
    malicious_clients = set(config.get("malicious_clients", []))
    attack_type = config.get("attack_type", "sign_flip")
    attack_strength = float(config.get("attack_strength", 1.0))
    enable_privacy = bool(config.get("enable_privacy", False))
    enable_dynamic_trust = bool(config.get("enable_dynamic_trust", True))

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load test data and metadata
    X_test, y_test = load_global_test_dataset()
    meta_df = pd.read_csv(os.path.join(CLIENT_DIR, "client_metadata.csv"))
    
    # Initialize client metadata state
    client_state = {}
    for _, row in meta_df.iterrows():
        cid = int(row["client_id"])
        client_state[cid] = {
            "client_id": cid,
            "trust": float(row["trust"]),
            "cpu_available": float(row["cpu_available"]),
            "memory_available": float(row["memory_available"]),
            "bandwidth": float(row["bandwidth"]),
            "latency_score": float(row["latency_score"]),
            "privacy_risk": float(row["privacy_risk"]),
            "loss": 1.0,
            "tprs": 0.5,
        }

    # Initialize Global Model
    global_model = MLP(input_dim=X_test.shape[1])
    global_params = get_torch_model_parameters(global_model)

    round_history = []
    start_time = time.time()

    for r in range(1, rounds + 1):
        # 1. Evaluate current TPRS or selection
        client_eval_list = list(client_state.values())
        
        if algorithm == "FedAvg":
            # FedAvg: random or sequential selection of clients_per_round
            # To be deterministic with seed:
            rng = np.random.default_rng(seed + r)
            selected_ids = list(rng.choice(num_clients, size=min(clients_per_round, num_clients), replace=False))
            for item in client_eval_list:
                item["selected"] = item["client_id"] in selected_ids
                item["tprs"] = 0.5
        else:
            # TPRS Selection
            ranked = rank_and_select_clients(client_eval_list, top_k=clients_per_round)
            for item in ranked:
                client_state[item["client_id"]] = item
            selected_ids = [c["client_id"] for c in ranked if c["selected"]]

        # 2. Local Training on Selected Clients
        selected_client_params = []
        selected_records = []

        for cid in selected_ids:
            local_params, sample_count, local_loss = local_train(
                client_id=cid,
                global_parameters=global_params,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                max_batches=max_batches,
            )
            # Update client's observed loss for next round quality
            client_state[cid]["loss"] = local_loss
            client_state[cid]["size"] = sample_count

            # Poisoning attack check
            is_mal = cid in malicious_clients
            if is_mal:
                local_params, _ = poison_parameters(
                    global_parameters=global_params,
                    local_parameters=local_params,
                    attack_type=attack_type,
                    attack_strength=attack_strength,
                )

            # Adaptive Privacy check
            if enable_privacy:
                local_params, _ = add_adaptive_noise(
                    global_parameters=global_params,
                    local_parameters=local_params,
                    privacy_risk=client_state[cid]["privacy_risk"],
                )

            selected_client_params.append(local_params)
            rec = dict(client_state[cid])
            rec["is_malicious"] = is_mal
            selected_records.append(rec)

        # 3. Dynamic Trust Update & Anomaly Mitigation (if enabled in TrustFed)
        if enable_dynamic_trust and algorithm != "FedAvg":
            # Compute reference update from benign average or median
            raw_updates = [[p - g for p, g in zip(c_p, global_params)] for c_p in selected_client_params]
            # Average update as rough consensus
            avg_update = [np.mean([u[li] for u in raw_updates], axis=0) for li in range(len(global_params))]
            for idx, cid in enumerate(selected_ids):
                sim = compute_update_cosine_similarity(avg_update, raw_updates[idx])
                # Update trust
                old_t = client_state[cid]["trust"]
                new_t = update_dynamic_trust(old_t, model_cosine_similarity=sim, loss_improvement=0.1)
                client_state[cid]["trust"] = new_t
                selected_records[idx]["trust"] = new_t
                # Recompute TPRS with new trust
                new_tprs = calculate_tprs(
                    trust=new_t,
                    quality=selected_records[idx]["quality"],
                    resource=selected_records[idx]["resource"],
                    network=selected_records[idx]["network"],
                    privacy_suitability=selected_records[idx]["privacy_suitability"],
                )
                selected_records[idx]["tprs"] = new_tprs
                client_state[cid]["tprs"] = new_tprs

        # 4. Aggregation Weight Computation
        method = "fedavg" if algorithm == "FedAvg" else "tprs"
        weights = compute_aggregation_weights(selected_records, method=method)
        for idx, rec in enumerate(selected_records):
            rec["aggregation_weight"] = float(weights[idx])

        # 5. Global Model Update
        aggregated_params = aggregate_model_parameters(selected_client_params, weights)
        set_torch_model_parameters(global_model, aggregated_params)
        global_params = aggregated_params

        # 6. Global Evaluation
        eval_metrics = evaluate_model(global_model, X_test, y_test)
        
        round_info = {
            "round": int(r),
            "selected_clients": [int(cid) for cid in selected_ids],
            "selected_malicious": [int(cid) for cid in selected_ids if cid in malicious_clients],
            "metrics": eval_metrics,
            "client_records": [
                {k: (int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v) for k, v in rec.items()}
                for rec in selected_records
            ],
            "weights": [float(w) for w in weights],
        }
        round_history.append(round_info)

        if progress_callback:
            progress_callback(r, rounds, eval_metrics)

    total_duration = time.time() - start_time
    final_metrics = round_history[-1]["metrics"]

    final_meta = []
    for c in client_state.values():
        clean_c = {k: (int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v) for k, v in c.items()}
        final_meta.append(clean_c)

    # Package output
    exp_output = {
        "experiment_id": exp_name,
        "config": config,
        "duration_seconds": round(float(total_duration), 2),
        "final_metrics": final_metrics,
        "round_history": round_history,
        "client_metadata_final": final_meta,
    }

    # Save to disk
    out_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(exp_output, f, indent=4)

    # Save model checkpoint
    torch.save(global_model.state_dict(), os.path.join(out_dir, "model_checkpoint.pth"))

    return exp_output
