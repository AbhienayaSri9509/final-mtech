"""
Security and Poisoning Attack Simulation Module for TrustFed-6G.

Threat Model:
    - Attackers: Byzantine edge clients (specified IDs, e.g., {0, 3, 7}).
    - Capability: Compromised clients can manipulate local model updates before aggregation.
    - Attacks:
        1. 'sign_flip': Update direction is inverted: Delta_adv = -gamma * Delta_clean.
        2. 'scaling': Update magnitude is amplified to dominate aggregation: Delta_adv = gamma * Delta_clean.
        3. 'gaussian': Random adversarial noise injected: Delta_adv = Delta_clean + N(0, gamma^2).
    - Defense Mechanism:
        - Cosine anomaly detection against global update direction.
        - Dynamic trust degradation (penalizing malicious updates).
        - Trust-weighted aggregation filtering.
"""

from typing import List, Set, Tuple, Dict, Any
import numpy as np


def is_malicious(client_id: int, malicious_clients: Set[int]) -> bool:
    """Check if client ID is in the designated compromised group."""
    return client_id in malicious_clients


def poison_parameters(
    global_parameters: List[np.ndarray],
    local_parameters: List[np.ndarray],
    attack_type: str = "sign_flip",
    attack_strength: float = 1.0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Apply a controlled parameter update poisoning attack.
    Returns:
        (poisoned_parameters, attack_metadata)
    """
    poisoned = []
    updates = [local - glob for glob, local in zip(global_parameters, local_parameters)]

    for global_param, update in zip(global_parameters, updates):
        if attack_type == "sign_flip":
            poisoned_update = -attack_strength * update
        elif attack_type == "scaling":
            poisoned_update = attack_strength * update
        elif attack_type == "gaussian":
            noise = np.random.normal(0, attack_strength, size=update.shape).astype(update.dtype)
            poisoned_update = update + noise
        else:
            raise ValueError(f"Unknown attack type: '{attack_type}'. Supported: 'sign_flip', 'scaling', 'gaussian'.")

        poisoned_parameter = global_param + poisoned_update
        poisoned.append(poisoned_parameter.astype(global_param.dtype))

    meta = {
        "attack_type": attack_type,
        "attack_strength": float(attack_strength),
        "is_poisoned": True,
    }
    return poisoned, meta


def compute_update_cosine_similarity(
    reference_update: List[np.ndarray],
    client_update: List[np.ndarray],
) -> float:
    """
    Compute cosine similarity between two multi-layer update vectors:
        cos(theta_ref, theta_client) = (u_ref . u_client) / (||u_ref|| * ||u_client||)
    """
    flat_ref = np.concatenate([p.flatten().astype(np.float64) for p in reference_update])
    flat_cli = np.concatenate([p.flatten().astype(np.float64) for p in client_update])

    norm_ref = np.linalg.norm(flat_ref)
    norm_cli = np.linalg.norm(flat_cli)

    if norm_ref == 0 or norm_cli == 0:
        return 0.0

    return float(np.dot(flat_ref, flat_cli) / (norm_ref * norm_cli))