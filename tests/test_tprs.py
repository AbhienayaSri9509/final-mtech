import pytest
import numpy as np
from trust.tprs import (
    calculate_quality_score,
    calculate_resource_score,
    calculate_network_score,
    calculate_privacy_suitability,
    calculate_tprs,
    rank_and_select_clients,
    update_dynamic_trust,
)

def test_tprs_formula_and_ranges():
    # Test valid calculation
    score = calculate_tprs(
        trust=0.8,
        quality=0.9,
        resource=0.7,
        network=0.6,
        privacy_suitability=0.5
    )
    # Expected: 0.30*0.8 + 0.25*0.9 + 0.20*0.7 + 0.15*0.6 + 0.10*0.5
    expected = 0.24 + 0.225 + 0.14 + 0.09 + 0.05
    assert abs(score - expected) < 1e-6
    assert 0.0 <= score <= 1.0

def test_quality_score_monotonicity():
    # Lower loss must give higher quality
    q_low_loss = calculate_quality_score(0.1)
    q_high_loss = calculate_quality_score(2.5)
    assert q_low_loss > q_high_loss
    assert calculate_quality_score(-0.5) <= 1.0
    assert calculate_quality_score(np.nan) == 1.0 or 0.0 <= calculate_quality_score(np.nan) <= 1.0

def test_network_and_resource_monotonicity():
    assert calculate_resource_score(0.9, 0.9) > calculate_resource_score(0.2, 0.2)
    assert calculate_network_score(0.8, 0.8) > calculate_network_score(0.3, 0.3)
    assert calculate_privacy_suitability(0.8) < calculate_privacy_suitability(0.2)

def test_client_ranking_and_top_7_selection():
    clients = [
        {"client_id": i, "trust": 0.5, "loss": 0.5 + i*0.1, "cpu_available": 0.5, "memory_available": 0.5, "bandwidth": 0.5, "latency_score": 0.5, "privacy_risk": 0.5}
        for i in range(10)
    ]
    ranked = rank_and_select_clients(clients, top_k=7)
    assert len(ranked) == 10
    selected = [c for c in ranked if c["selected"]]
    assert len(selected) == 7
    # Client 0 had lowest loss, so highest quality, should be selected and first
    assert ranked[0]["client_id"] == 0
    assert ranked[0]["selected"] is True

def test_dynamic_trust_updating():
    initial_trust = 0.5
    # Malicious or opposite update (cosine similarity -0.9)
    penalized = update_dynamic_trust(initial_trust, model_cosine_similarity=-0.9, loss_improvement=-0.5)
    assert penalized < initial_trust

    # Highly cooperative update (cosine similarity +0.95, positive loss improvement)
    rewarded = update_dynamic_trust(initial_trust, model_cosine_similarity=0.95, loss_improvement=0.2)
    assert rewarded > initial_trust
