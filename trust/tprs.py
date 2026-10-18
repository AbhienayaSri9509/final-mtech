"""
Trust-Privacy-Resource Score (TPRS) Module for TrustFed-6G.

TPRS Formulation:
    TPRS_i = w_T * T_i + w_Q * Q_i + w_R * R_i + w_C * C_i + w_S * S_i

Default Weights:
    w_T = 0.30 (Client Trust Score)
    w_Q = 0.25 (Local Model/Data Quality: 1 / (1 + loss))
    w_R = 0.20 (Resource Availability: 0.5 * CPU + 0.5 * RAM)
    w_C = 0.15 (Network Suitability: 0.5 * Bandwidth + 0.5 * LatencyScore)
    w_S = 0.10 (Privacy Suitability: 1 - PrivacyRisk)
"""

from typing import List, Dict, Any, Optional
import numpy as np


DEFAULT_WEIGHTS = {
    "trust": 0.30,
    "quality": 0.25,
    "resource": 0.20,
    "network": 0.15,
    "privacy": 0.10,
}


def validate_and_normalize(value: Any, name: str, default: float = 0.5) -> float:
    """Validate that score component is numeric, finite, and clamped to [0.0, 1.0]."""
    if value is None:
        return default
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return default
        return float(np.clip(val, 0.0, 1.0))
    except (TypeError, ValueError):
        return default


def calculate_quality_score(loss: float) -> float:
    """
    Compute local model quality score Q_i in [0.0, 1.0].
    Lower local training loss yields higher quality: Q = 1 / (1 + max(0, loss)).
    """
    try:
        val = float(loss)
        if np.isnan(val) or np.isinf(val) or val < 0:
            val = 0.0
        return float(np.clip(1.0 / (1.0 + val), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.5


def calculate_resource_score(cpu_avail: float, mem_avail: float) -> float:
    """R_i = 0.5 * cpu_available + 0.5 * memory_available."""
    cpu = validate_and_normalize(cpu_avail, "cpu")
    mem = validate_and_normalize(mem_avail, "memory")
    return float(np.clip(0.5 * cpu + 0.5 * mem, 0.0, 1.0))


def calculate_network_score(bandwidth: float, latency_score: float) -> float:
    """C_i = 0.5 * bandwidth + 0.5 * latency_score."""
    bw = validate_and_normalize(bandwidth, "bandwidth")
    lat = validate_and_normalize(latency_score, "latency")
    return float(np.clip(0.5 * bw + 0.5 * lat, 0.0, 1.0))


def calculate_privacy_suitability(privacy_risk: float) -> float:
    """S_i = 1 - privacy_risk."""
    risk = validate_and_normalize(privacy_risk, "privacy_risk")
    return float(np.clip(1.0 - risk, 0.0, 1.0))


def calculate_tprs(
    trust: float,
    quality: float,
    resource: float,
    network: float,
    privacy_suitability: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate composite TPRS score in [0.0, 1.0].
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    t = validate_and_normalize(trust, "trust")
    q = validate_and_normalize(quality, "quality")
    r = validate_and_normalize(resource, "resource")
    c = validate_and_normalize(network, "network")
    s = validate_and_normalize(privacy_suitability, "privacy")

    score = (
        w.get("trust", 0.30) * t
        + w.get("quality", 0.25) * q
        + w.get("resource", 0.20) * r
        + w.get("network", 0.15) * c
        + w.get("privacy", 0.10) * s
    )
    return float(np.clip(score, 0.0, 1.0))


def rank_and_select_clients(
    client_records: List[Dict[str, Any]],
    top_k: int = 7,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute TPRS for all eligible clients, rank deterministically,
    and select top-k clients.
    
    Tie-breaking: (tprs DESC, trust DESC, client_id ASC).
    """
    evaluated = []
    for record in client_records:
        cid = record.get("client_id")
        trust = validate_and_normalize(record.get("trust", 0.5), "trust")
        
        # Local loss / quality
        if "quality" in record:
            quality = validate_and_normalize(record["quality"], "quality")
        else:
            loss = record.get("loss", 1.0)
            quality = calculate_quality_score(loss)

        resource = calculate_resource_score(
            record.get("cpu_available", 0.5), record.get("memory_available", 0.5)
        )
        network = calculate_network_score(
            record.get("bandwidth", 0.5), record.get("latency_score", 0.5)
        )
        priv_suit = calculate_privacy_suitability(record.get("privacy_risk", 0.5))

        tprs = calculate_tprs(
            trust=trust,
            quality=quality,
            resource=resource,
            network=network,
            privacy_suitability=priv_suit,
            weights=weights,
        )

        entry = dict(record)
        entry["trust"] = trust
        entry["quality"] = quality
        entry["resource"] = resource
        entry["network"] = network
        entry["privacy_suitability"] = priv_suit
        entry["tprs"] = tprs
        entry["selected"] = False
        evaluated.append(entry)

    # Sort deterministically
    evaluated.sort(
        key=lambda x: (
            -round(x["tprs"], 6),
            -round(x["trust"], 6),
            int(x.get("client_id", 0)),
        )
    )

    # Mark top-k selected
    for i, item in enumerate(evaluated):
        if i < top_k:
            item["selected"] = True

    return evaluated


def update_dynamic_trust(
    current_trust: float,
    model_cosine_similarity: float,
    loss_improvement: float,
    alpha: float = 0.20,
) -> float:
    """
    Dynamic Trust Update Extension:
    Adjusts client trust based on historical behavior and update quality.
    
    T_{t+1} = (1 - alpha) * T_t + alpha * update_credibility
    where update_credibility is derived from directional alignment (cosine similarity)
    and positive contribution to loss minimization.
    """
    # Normalize cosine similarity from [-1, 1] to [0, 1]
    cos_norm = float(np.clip((model_cosine_similarity + 1.0) / 2.0, 0.0, 1.0))
    # Loss credibility
    loss_score = 1.0 if loss_improvement >= 0 else float(np.clip(1.0 / (1.0 + abs(loss_improvement)), 0.0, 1.0))
    credibility = 0.70 * cos_norm + 0.30 * loss_score

    new_trust = (1.0 - alpha) * current_trust + alpha * credibility
    return float(np.clip(new_trust, 0.05, 1.0))