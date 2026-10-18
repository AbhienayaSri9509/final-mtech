# TrustFed-6G: Adaptive Privacy-Aware Federated Learning for Secure and Resource-Efficient 6G Edge Networks

An end-to-end research prototype and interactive simulation framework for M.Tech research on Trust-Privacy-Resource Scored (TPRS) Federated Learning in 6G edge intrusion detection scenarios.

---

## 1. Quick Start (Windows PowerShell)

Ensure Python 3.10+ is installed on your Windows machine:

```powershell
# 1. Navigate to the project directory
cd C:\Users\HP\Downloads\TrustFed-AI-main\TrustFed-AI-main

# 2. Install required dependencies
python -m pip install -r requirements.txt
python -m pip install streamlit plotly

# 3. Partition UNSW-NB15 dataset into 10 client splits (if not already done)
python -m clients.create_clients

# 4. Run automated test suite
python -m pytest tests/

# 5. Run the reproducible benchmark suite (Experiments A-E)
python -m scripts.run_benchmarks

# 6. Launch the interactive research web dashboard
streamlit run app.py
```

---

## 2. Benchmark Experimental Results

Evaluated on the held-out global UNSW-NB15 test set (35,069 records):

| Experiment | Algorithm | Conditions | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **`exp_a_fedavg_benign`** | FedAvg | Clean / No Attack | **93.95%** | 92.93% | 98.61% | **95.69%** |
| **`exp_b_fedavg_attacked`** | FedAvg | Byzantine Clients {0,3,7} Sign-Flip | **93.21%** | 91.31% | 99.50% | **95.23%** |
| **`exp_c_trustfed_attacked`** | TrustFed-6G | Byzantine Clients {0,3,7} + Defense | **93.97%** | 93.13% | 98.40% | **95.69%** |
| **`exp_d_trustfed_privacy`** | TrustFed-6G | Adaptive DP Perturbation | **92.84%** | 90.77% | 99.61% | **94.98%** |
| **`exp_e_trustfed_clean`** | TrustFed-6G | Clean / Full TPRS Selection | **93.97%** | 93.13% | 98.40% | **95.69%** |

---

## 3. Project Architecture

- **`trust/tprs.py`**: TPRS scoring formula, component normalization, deterministic top-7 ranking, and dynamic trust updates.
- **`server/trust_aware_aggregation.py`**: Parameter validation and TPRS weighted aggregation:
  $$\alpha_i = \frac{\text{TPRS}_i \cdot n_i}{\sum_j \text{TPRS}_j \cdot n_j}$$
- **`server/experiment_runner.py`**: Unified orchestration engine saving reproducible runs to `results/experiments/<id>/`.
- **`privacy/adaptive_dp.py`**: Update L2-norm clipping and calibrated Gaussian perturbation based on client privacy-risk tiers.
- **`security/poisoning.py`**: Sign-flip poisoning attack simulation and cosine similarity anomaly detection.
- **`app.py`**: 6-page interactive Streamlit dashboard.
- **`docs/`**: Comprehensive guides, audit report, experiment protocol, and viva defense notes.

---

## 4. Documentation Directory

- [Implementation Audit](docs/IMPLEMENTATION_AUDIT.md)
- [Implementation & Architecture Guide](docs/IMPLEMENTATION_GUIDE.md)
- [Experiment Protocol & Benchmark Results](docs/EXPERIMENT_PROTOCOL.md)
- [Results Interpretation & Viva Q&A Guide](docs/RESULTS_INTERPRETATION.md)
- [Limitations & Research Boundaries](docs/LIMITATIONS.md)
- [Website User Guide](docs/WEBSITE_USER_GUIDE.md)

---

## 5. Disclaimer

*This is a software-based edge simulation on the UNSW-NB15 dataset. It does not deploy real physical 6G base stations or physical IoT hardware.*
