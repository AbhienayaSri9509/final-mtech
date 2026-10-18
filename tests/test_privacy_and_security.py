import pytest
import numpy as np
from privacy.adaptive_dp import (
    get_privacy_epsilon,
    clip_update_l2_norm,
    add_adaptive_noise,
)
from security.poisoning import (
    is_malicious,
    poison_parameters,
    compute_update_cosine_similarity,
)

def test_privacy_epsilon_mapping():
    assert get_privacy_epsilon(0.85) == 0.5
    assert get_privacy_epsilon(0.55) == 1.0
    assert get_privacy_epsilon(0.20) == 2.0

def test_l2_clipping():
    # Large update vector
    update = [np.ones((10, 10), dtype=np.float32) * 5.0]  # norm = sqrt(100 * 25) = 50.0
    clipped, raw_norm = clip_update_l2_norm(update, max_norm=5.0)
    assert abs(raw_norm - 50.0) < 1e-4
    clipped_norm = np.linalg.norm(clipped[0])
    assert abs(clipped_norm - 5.0) < 1e-4

def test_adaptive_noise_injection():
    glob = [np.zeros((10, 10), dtype=np.float32)]
    loc = [np.ones((10, 10), dtype=np.float32)]
    noisy, info = add_adaptive_noise(glob, loc, privacy_risk=0.8, clip_threshold=1.0)
    assert info["epsilon"] == 0.5
    assert not np.allclose(noisy[0], loc[0])
    assert not np.isnan(noisy[0]).any()

def test_sign_flip_poisoning():
    glob = [np.array([1.0, 2.0], dtype=np.float32)]
    loc = [np.array([1.5, 2.5], dtype=np.float32)]  # clean update = [+0.5, +0.5]
    poisoned, meta = poison_parameters(glob, loc, attack_type="sign_flip", attack_strength=1.0)
    # Poisoned update should be -1.0 * [+0.5, +0.5] = [-0.5, -0.5]
    # Poisoned param = glob + update = [0.5, 1.5]
    np.testing.assert_allclose(poisoned[0], np.array([0.5, 1.5], dtype=np.float32))

def test_cosine_similarity_detection():
    clean_update = [np.array([1.0, 1.0])]
    flipped_update = [np.array([-1.0, -1.0])]
    sim = compute_update_cosine_similarity(clean_update, flipped_update)
    assert abs(sim - (-1.0)) < 1e-5
