# TrustFed-6G Implementation & Architecture Guide

## 1. System Overview
**TrustFed-6G** is an adaptive, privacy-aware federated learning prototype designed for secure and resource-efficient intrusion detection in simulated 6G edge environments.

The architecture addresses three core challenges of distributed edge networks:
1. **Device & Network Heterogeneity:** Dynamic multi-criteria client selection balancing trust, model loss, compute, memory, and bandwidth.
2. **Adversarial Poisoning Vulnerability:** Robust mitigation against byzantine model update poisoning (such as sign-flip attacks).
3. **Privacy Exposure:** Calibrated adaptive update perturbation mapped to client privacy-risk tiers.

```
       +-------------------------------------------------------------+
       |                  TRUSTFED-6G SERVER ORCHESTRATION           |
       |  - TPRS Ranking & Top-K Selection                           |
       |  - Dynamic Consensus Anomaly Detection                      |
       |  - Weighted Model Aggregation: alpha_i = (TPRS_i * n_i)/sum |
       |  - Global Evaluation on Held-Out UNSW-NB15 Benchmark         |
       +------------------------------+------------------------------+
                                      |
                     Dispatch Global Weights & Collect Updates
                                      |
       +------------------------------v------------------------------+
       |                  SIMULATED 6G EDGE CLIENTS (0 - 9)          |
       |  - Local Training on Disjoint Partitions (14,027 samples)   |
       |  - Adaptive Privacy Perturbation (L2-norm clip + Gaussian)  |
       |  - Byzantine Adversaries ({0, 3, 7} Sign-Flip Poisoning)    |
       |  - Resource & Network Telemetry (CPU, RAM, BW, Latency)     |
       +-------------------------------------------------------------+
```

---

## 2. Directory Layout & Module Responsibilities

- **`model/`**:
  - `mlp_model.py`: 4-layer PyTorch MLP (192 input $\to$ 128 $\to$ 64 $\to$ 32 $\to$ 1 output logit) with BCEWithLogitsLoss.
  - `train_baseline.py`: Centralized training benchmark.
- **`clients/`**:
  - `create_clients.py`: Stratified preprocessing and partitioning of UNSW-NB15 into 10 client `.npz` arrays and `global_test.npz`.
- **`trust/`**:
  - `tprs.py`: Mathematical implementation of TPRS scoring, component normalization, deterministic top-7 selection, and dynamic trust updates.
- **`server/`**:
  - `trust_aware_aggregation.py`: Validation of parameter tensor compatibility, normalized weight calculation ($\alpha_i$), and PyTorch model loading.
  - `experiment_runner.py`: Unified pipeline orchestrating clean, attacked, privacy, and ablation runs.
- **`privacy/`**:
  - `adaptive_dp.py`: Sensitivity clipping (L2-norm bound) and Gaussian noise perturbation calibrated to privacy-risk categories.
- **`security/`**:
  - `poisoning.py`: Sign-flip, scaling, and Gaussian weight poisoning attack simulation with cosine similarity anomaly detection.
- **`app.py`**:
  - 6-page interactive Streamlit research web dashboard.
- **`tests/`**:
  - Comprehensive automated test suite (17 passed unit & integration tests).

---

## 3. TPRS Mathematical Formulation

$$\text{TPRS}_i = 0.30 \cdot T_i + 0.25 \cdot Q_i + 0.20 \cdot R_i + 0.15 \cdot C_i + 0.10 \cdot S_i$$

Where:
- $T_i$: Client trust score in $[0, 1]$. Initialized to neutral (0.50), updated dynamically based on consensus cosine similarity.
- $Q_i = \frac{1}{1 + \text{loss}_i}$: Local model quality score.
- $R_i = 0.5 \cdot \text{cpu}_i + 0.5 \cdot \text{mem}_i$: Resource score.
- $C_i = 0.5 \cdot \text{bandwidth}_i + 0.5 \cdot \text{latency}_i$: Network suitability score.
- $S_i = 1 - \text{privacy\_risk}_i$: Privacy suitability score.

Aggregation Weighting:
$$\alpha_i = \frac{\text{TPRS}_i \cdot n_i}{\sum_{j \in \mathcal{S}} \text{TPRS}_j \cdot n_j}$$
All weights $\alpha_i \ge 0$ and $\sum_{i \in \mathcal{S}} \alpha_i = 1.0$.
