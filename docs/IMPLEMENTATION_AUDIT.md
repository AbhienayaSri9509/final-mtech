# TrustFed-6G Implementation Audit

**Date:** 2026-10-18  
**Project:** TrustFed-6G: Adaptive Privacy-Aware Federated Learning for Secure and Resource-Efficient 6G Edge Networks  
**Workspace Root:** `c:\Users\HP\Downloads\TrustFed-AI-main\TrustFed-AI-main`  
**Auditor:** Senior ML/FL Research & Systems Engineer  

---

## 1. Executive Summary

An exhaustive inspection of the project workspace was conducted across code files, dataset structures, existing pre-trained checkpoints, results, documentation, and dependencies. The workspace contains a substantial foundational codebase with key FL simulation components (FedAvg, TPRS, differential privacy noise, and sign-flip poisoning attack scripts), along with pre-generated ablation models and figures. However, critical gaps, hardcoded assumptions, discrepancies between scripts, and missing modules (notably an interactive research web dashboard, dynamic trust calculation, unified experiment orchestration, and data-split caching) require systematic implementation and remediation.

---

## 2. Directory Structure & File Inventory

| Path | Size / Lines | Status | Purpose & Notes |
|---|---|---|---|
| `data/raw/UNSW_NB15_training-set.xlsx` | 42.3 MB (175,341 × 45) | Working / Intact | Primary raw training dataset. Contains binary `label`, `id`, `attack_cat`, and 42 network features. |
| `data/raw/UNSW_NB15_testing-set.xlsx` | 19.9 MB (82,332 × 45) | Working / Intact | Standard testing set from UNSW-NB15 benchmark. |
| `data/raw/UNSW_NB15_features.csv` | 4.0 KB | Working / Intact | Feature descriptions and metadata for UNSW-NB15. |
| `data/client_splits/` | Empty (`.gitkeep`) | Missing Data | Intended directory for partitioned client `.npz` files and `global_test.npz`. Currently empty. |
| `data/processed/` | Empty (`.gitkeep`) | Empty | Processed full datasets cache. |
| `clients/create_clients.py` | 461 lines | Needs Execution & Fix | Partitions training data into 10 client splits with metadata (`client_metadata.csv`). Takes ~40-60s to run due to Excel parsing. |
| `configs/config.json` | 23 lines | Partial | Contains baseline configuration (10 clients, 5 rounds/clients_per_round, learning rate, TPRS weights). |
| `model/mlp_model.py` | 10 lines | Discrepancy | Contains a simple 2-output Linear(64, num_classes) MLP. **Discrepancy:** The actual trained baseline and server scripts use a binary 1-output (`Linear(32, 1)`) MLP architecture with BCEWithLogitsLoss. |
| `model/train_baseline.py` | 575 lines | Complete / Runnable | Centralized MLP baseline pipeline on UNSW-NB15. Fits ColumnTransformer (StandardScaler + OneHotEncoder -> 192 features). |
| `model/data_loader.py` | Minimal | Redundant | Bare helper; data loading is directly handled in scripts. |
| `trust/tprs.py` | 48 lines | Working / Needs Extension | Implements `calculate_tprs(trust, quality, resource, network, privacy_suitability)` and `rank_clients`. Lacks input validation, clipping, and tie-breaking. |
| `trust/trust_score.py` | 253 bytes | Stub | Minimal stub. |
| `server/trust_aware_aggregation.py` | 9 lines | Incomplete / Stub | Bare stub with flawed formulas (`normalized_tprs_weights` uses unnormalized product). Needs full mathematical implementation: $\alpha_i = \frac{\text{TPRS}_i \cdot n_i}{\sum_j \text{TPRS}_j \cdot n_j}$. |
| `server/fedavg_server.py` | 764 lines | Complete simulation | Standard FedAvg baseline server (5 rounds, 10 clients, BCEWithLogitsLoss). |
| `server/tprs_server.py` | 1,494 lines | Complete simulation | TPRS client selection and aggregation server. |
| `server/trustfed_privacy_server.py` | 1,362 lines | Complete simulation | TPRS with adaptive Gaussian noise based on client privacy risk. |
| `server/trustfed_attack_server.py` | 1,320 lines | Complete simulation | Poisoning attack experiment with malicious clients {0, 3, 7} and sign-flip. |
| `server/ablation_server.py` | 1,135 lines | Complete simulation | Full ablation runner (A1 to A6). |
| `privacy/adaptive_dp.py` | 99 lines | Functional experimental noise | Maps privacy risk $\ge 0.7 \to \epsilon=0.5$, $\ge 0.4 \to \epsilon=1.0$, $< 0.4 \to \epsilon=2.0$. Adds Gaussian noise to weights. |
| `security/poisoning.py` | 87 lines | Working | Implements sign_flip, scaling, and gaussian weight poisoning. |
| `results/figures/` | 5 PNGs + 1 CSV | Historical artifacts | Contains `final_ablation_table.csv` and matplotlib ablation plots. |
| `results/models/` | 11 `.pth` checkpoints | Existing artifacts | Contains centralized and federated saved models (192 input dim, hidden 128->64->32->1). |
| `results/metrics/` | Empty (`.gitkeep`) | Incomplete | Per-experiment JSON metrics were not committed or saved in a unified schema. |
| `results/generate_graphs.py` | 368 lines | Runnable | Generates ablation comparison figures from `results/metrics/ablation_results.csv`. |
| `docs/` | 3 Markdown files | Outdated / Incomplete | High-level notes from earlier planning. |

---

## 3. Dataset & Preprocessing Details

- **Raw Data Source:** UNSW-NB15 dataset (`data/raw/UNSW_NB15_training-set.xlsx` with 175,341 rows, 45 columns; `data/raw/UNSW_NB15_testing-set.xlsx` with 82,332 rows).
- **Target Variable:** `label` (binary classification: 0 = Normal, 1 = Attack).
- **Leakage Prevention:**
  - `attack_cat` (multi-class string) is dropped to prevent leakage.
  - `id` column is dropped (row identifier).
  - Categorical columns (`proto`, `service`, `state`) are one-hot encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - Numerical columns (39 continuous/discrete counts and rates) are imputed with median and scaled using `StandardScaler`.
  - The pipeline (`ColumnTransformer`) is fit strictly on training splits and applied to test sets.
  - Output feature dimension is verified to be exactly **192**.
- **Issue Discovered:** The raw data is stored as `.xlsx` (Excel), which requires ~45 seconds to parse each time. Generating and saving fast-loading `.npz` splits in `data/client_splits/` (and saving preprocessed CSV/Parquet) will provide 100x speedup for iterative experiments and web execution.

---

## 4. Model Architecture & Training Verification

- **Actual Checkpoint Architecture:**
  - Input: 192 features
  - Layer 1: Linear(192, 128) -> ReLU -> BatchNorm1d(128) (in centralized baseline) or Dropout / ReLU
  - Layer 2: Linear(128, 64) -> ReLU
  - Layer 3: Linear(64, 32) -> ReLU
  - Layer 4: Linear(32, 1) -> Output Logit
  - Loss: `BCEWithLogitsLoss`
  - Optimizer: `Adam(lr=0.001)`
- **Architectural Reconciliation:**
  - `model/mlp_model.py` had a mismatch (2 output classes with CrossEntropy). It must be aligned with `server/*.py` and `results/models/*.pth` (single logit with Sigmoid/BCEWithLogitsLoss) for full consistency across all modules.

---

## 5. Mathematical & Algorithmic Audit

### 5.1 TPRS Calculation
$$\text{TPRS}_i = 0.30 \cdot T_i + 0.25 \cdot Q_i + 0.20 \cdot R_i + 0.15 \cdot C_i + 0.10 \cdot S_i$$
- $T_i$: Client trust score $\in [0, 1]$.
- $Q_i = \frac{1}{1 + \text{loss}_i}$: Local model quality score.
- $R_i = 0.5 \cdot \text{cpu}_i + 0.5 \cdot \text{mem}_i$: Resource suitability score.
- $C_i = 0.5 \cdot \text{bandwidth}_i + 0.5 \cdot \text{latency}_i$: Network suitability score.
- $S_i = 1 - \text{privacy\_risk}_i$: Privacy suitability score.

**Audit Findings:**
- In the initial code, $T_i$ is initialized to 0.50 and remains static throughout training rounds.
- As required by the research guidelines: static trust will be evaluated as the initial baseline, and a **dynamic trust update mechanism** (evaluating model update cosine distance / loss trajectory to adjust trust) will be implemented as a separately evaluated extension.

### 5.2 Aggregation Weighting
$$\alpha_i = \frac{\text{TPRS}_i \cdot n_i}{\sum_{j \in \mathcal{S}} \text{TPRS}_j \cdot n_j}$$
- Where $\mathcal{S}$ is the set of selected clients in round $t$, and $n_i$ is client $i$'s dataset size.
- **Audit Findings:** In `server/tprs_server.py`, the aggregation implements this formula directly. In `server/trust_aware_aggregation.py`, the code is an outdated 9-line stub. This will be centralized into a robust, tested aggregation module.

### 5.3 Privacy Implementation
- Implemented as parameter-level Gaussian noise scaled by $\frac{\text{sensitivity}}{\epsilon} \cdot 0.01$.
- **Audit Findings:** This is an *experimental heuristic noise mechanism* and does not constitute formal Rényi or moments differential privacy. Per the instructions, this will be transparently documented and labeled as an experimental perturbation mechanism rather than fabricating formal DP guarantees.

### 5.4 Poisoning Attack Implementation
- Controlled sign-flip attack: $\theta_{\text{poisoned}} = \theta_{\text{global}} - \gamma (\theta_{\text{local}} - \theta_{\text{global}})$ with $\gamma = 1.0$ applied by clients $\{0, 3, 7\}$.
- **Audit Findings:** The attack logic in `security/poisoning.py` correctly implements update inversion. An anomaly-based trust degradation mechanism will be incorporated to evaluate defense effectiveness.

---

## 6. Dependencies & Environment Status

- **Python Version:** Python 3.14.7 (Windows x64).
- **Core ML Packages Installed:** `torch` (2.14.1), `numpy` (2.5.3), `pandas` (3.0.6), `scikit-learn` (1.9.1), `openpyxl` (3.1.5), `matplotlib` (3.11.2).
- **Missing Dashboard Packages:** `streamlit` and `plotly` (installation launched via pip in background).
- **Execution Script:** `run_first_steps.bat` exists for basic execution.

---

## 7. Action Plan & Next Implementation Tasks

1. **Stage 1 (Data & Baseline Validation):** Run `clients/create_clients.py` to generate real partitioned datasets in `data/client_splits/` (`client_0.npz` to `client_9.npz`, `global_test.npz`, `client_metadata.csv`). Verify dimensions (192 features, binary labels). Fix `model/mlp_model.py`.
2. **Stage 2 (TPRS Module):** Refactor `trust/tprs.py` with strict input validation, score component logging, deterministic ranking, top-7 selection, and dynamic trust updates. Add unit tests.
3. **Stage 3 (Trust-Aware Aggregation):** Implement `trust/aggregation.py` with validation of parameter shapes, weight normalization, zero-denominator safety, and fallback handling.
4. **Stage 4 (Privacy Simulation):** Refactor `privacy/adaptive_dp.py` with clipping, parameter noise tracking, and empirical privacy-utility metrics.
5. **Stage 5 (Poisoning Attack & Defense):** Unify attack simulation with Experiments A, B, and C (Benign FedAvg, Attacked FedAvg, Attacked TrustFed with dynamic trust penalty).
6. **Stage 6 (Experiment Pipeline):** Implement `server/experiment_runner.py` allowing JSON/CLI execution saving to `results/experiments/<id>/` with full per-round metrics, weights, and confusion matrices.
7. **Stage 7 (Fair Evaluation):** Execute reproducible benchmarks across all configurations and store unified JSON/CSV metrics.
8. **Stage 8 & 9 (Streamlit Research Dashboard):** Build 6-page interactive Streamlit dashboard (`app.py`) reading actual experiment results, featuring live experiment execution with progress, Plotly charts, and client inspection.
9. **Stage 10 (Testing & Documentation):** Write comprehensive test suite (`tests/`) and produce all requested documentation (`docs/IMPLEMENTATION_GUIDE.md`, etc.).
