"""
Benchmark Execution Script for TrustFed-6G Comparative Study.

Executes and archives standard reproducible benchmark runs:
1. exp_a_fedavg_benign (Standard FedAvg, no attacks)
2. exp_b_fedavg_attacked (Standard FedAvg under {0,3,7} sign-flip attack)
3. exp_c_trustfed_attacked (TrustFed-6G with TPRS and Dynamic Trust defense)
4. exp_d_trustfed_privacy (TrustFed-6G with Adaptive DP noise)
5. exp_e_trustfed_clean (TrustFed-6G under clean conditions)
"""

import os
import json
from server.experiment_runner import run_experiment

BENCHMARKS = [
    {
        "name": "exp_a_fedavg_benign",
        "algorithm": "FedAvg",
        "rounds": 5,
        "clients_per_round": 7,
        "num_clients": 10,
        "local_epochs": 1,
        "batch_size": 256,
        "learning_rate": 0.001,
        "seed": 42,
        "malicious_clients": [],
        "enable_privacy": False,
        "enable_dynamic_trust": False,
        "description": "Experiment A: Standard FedAvg under benign conditions"
    },
    {
        "name": "exp_b_fedavg_attacked",
        "algorithm": "FedAvg",
        "rounds": 5,
        "clients_per_round": 7,
        "num_clients": 10,
        "local_epochs": 1,
        "batch_size": 256,
        "learning_rate": 0.001,
        "seed": 42,
        "malicious_clients": [0, 3, 7],
        "attack_type": "sign_flip",
        "attack_strength": 1.0,
        "enable_privacy": False,
        "enable_dynamic_trust": False,
        "description": "Experiment B: Standard FedAvg under sign-flip poisoning attack"
    },
    {
        "name": "exp_c_trustfed_attacked",
        "algorithm": "TrustFed-6G",
        "rounds": 5,
        "clients_per_round": 7,
        "num_clients": 10,
        "local_epochs": 1,
        "batch_size": 256,
        "learning_rate": 0.001,
        "seed": 42,
        "malicious_clients": [0, 3, 7],
        "attack_type": "sign_flip",
        "attack_strength": 1.0,
        "enable_privacy": False,
        "enable_dynamic_trust": True,
        "description": "Experiment C: TrustFed-6G defense with TPRS and Dynamic Trust"
    },
    {
        "name": "exp_d_trustfed_privacy",
        "algorithm": "TrustFed-6G",
        "rounds": 5,
        "clients_per_round": 7,
        "num_clients": 10,
        "local_epochs": 1,
        "batch_size": 256,
        "learning_rate": 0.001,
        "seed": 42,
        "malicious_clients": [],
        "enable_privacy": True,
        "enable_dynamic_trust": True,
        "description": "Experiment D: TrustFed-6G with Adaptive Differential Privacy Noise"
    },
    {
        "name": "exp_e_trustfed_clean",
        "algorithm": "TrustFed-6G",
        "rounds": 5,
        "clients_per_round": 7,
        "num_clients": 10,
        "local_epochs": 1,
        "batch_size": 256,
        "learning_rate": 0.001,
        "seed": 42,
        "malicious_clients": [],
        "enable_privacy": False,
        "enable_dynamic_trust": True,
        "description": "Experiment E: Full TrustFed-6G under benign conditions"
    }
]

def run_all_benchmarks():
    print("=" * 70)
    print("STARTING TRUSTFED-6G BENCHMARK SUITE")
    print("=" * 70)
    summary = []
    for cfg in BENCHMARKS:
        print(f"\n---> Running: {cfg['name']} ({cfg['description']})")
        res = run_experiment(cfg)
        metrics = res["final_metrics"]
        print(f"Finished {cfg['name']} in {res['duration_seconds']}s")
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}")
        summary.append({
            "name": cfg["name"],
            "algorithm": cfg["algorithm"],
            "description": cfg["description"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "duration": res["duration_seconds"],
        })
    
    # Save overall summary
    summary_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics", "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print("\nBenchmark suite complete. Saved summary to:", summary_path)

if __name__ == "__main__":
    run_all_benchmarks()
