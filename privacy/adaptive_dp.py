"""
Adaptive Privacy and Experimental Perturbation Mechanism for TrustFed-6G.

Specification:
    - Privacy Unit: Per-client parameter update vector (Delta theta_i = theta_local,i - theta_global)
    - Sensitivity Bound / Clipping: L2-norm clipping of local updates:
        Delta theta_clipped = Delta theta / max(1.0, ||Delta theta||_2 / clip_threshold)
    - Privacy Risk Mapping:
        Risk >= 0.70 -> epsilon = 0.5 (Strong privacy, higher noise)
        0.40 <= Risk < 0.70 -> epsilon = 1.0 (Moderate privacy)
        Risk < 0.40 -> epsilon = 2.0 (Low privacy, minimal noise)
    - Noise Mechanism: Gaussian perturbation calibrated to clip threshold and epsilon:
        sigma = (clip_threshold / epsilon) * noise_multiplier
        Delta theta_noisy = Delta theta_clipped + N(0, sigma^2)
        theta_local_private = theta_global + Delta theta_noisy

Note: This is an experimental client-level privacy perturbation simulation.
It does not replace a cryptographically verified RDP/moments-accountant bound across infinite queries.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


def get_privacy_epsilon(privacy_risk: float) -> float:
    """
    Map client privacy risk to experimental epsilon.
    Higher risk -> smaller epsilon -> higher noise level.
    """
    risk = float(np.clip(privacy_risk, 0.0, 1.0))
    if risk >= 0.70:
        return 0.5
    elif risk >= 0.40:
        return 1.0
    else:
        return 2.0


def clip_update_l2_norm(
    update_params: List[np.ndarray],
    max_norm: float = 1.0,
) -> Tuple[List[np.ndarray], float]:
    """
    Clip update vector to a maximum L2 sensitivity bound:
        C = max_norm
        scaling = min(1.0, C / ||Delta||_2)
    """
    total_norm = np.sqrt(sum(np.sum(p.astype(np.float64) ** 2) for p in update_params))
    scaling = 1.0
    if total_norm > max_norm and total_norm > 0:
        scaling = max_norm / float(total_norm)

    clipped = [p * scaling for p in update_params]
    return clipped, float(total_norm)


def add_adaptive_noise(
    global_parameters: List[np.ndarray],
    local_parameters: List[np.ndarray],
    privacy_risk: float,
    clip_threshold: float = 1.0,
    noise_multiplier: float = 0.01,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Apply clipped adaptive perturbation to the local client update.
    Returns:
        (private_local_parameters, privacy_metrics_dict)
    """
    epsilon = get_privacy_epsilon(privacy_risk)
    
    # 1. Compute update
    updates = [local - glob for glob, local in zip(global_parameters, local_parameters)]
    
    # 2. Clip update
    clipped_updates, raw_l2_norm = clip_update_l2_norm(updates, max_norm=clip_threshold)
    
    # 3. Calibrate noise
    sigma = (clip_threshold / epsilon) * noise_multiplier
    
    noisy_updates = []
    total_noise_norm = 0.0
    for u in clipped_updates:
        noise = np.random.normal(loc=0.0, scale=sigma, size=u.shape).astype(u.dtype)
        total_noise_norm += float(np.sum(noise.astype(np.float64) ** 2))
        noisy_updates.append(u + noise)
    
    # 4. Reconstruct private parameters
    private_parameters = [glob + noisy_u for glob, noisy_u in zip(global_parameters, noisy_updates)]

    info = {
        "privacy_risk": float(privacy_risk),
        "epsilon": float(epsilon),
        "clip_threshold": float(clip_threshold),
        "raw_l2_norm": float(raw_l2_norm),
        "sigma": float(sigma),
        "noise_l2_norm": float(np.sqrt(total_noise_norm)),
    }
    return private_parameters, info