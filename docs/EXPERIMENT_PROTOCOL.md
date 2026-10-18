# TrustFed-6G Experiment Protocol & Evaluation Benchmark

## 1. Experimental Setup

- **Dataset:** UNSW-NB15 Network Intrusion Detection Dataset.
- **Samples:** 140,272 training samples partitioned across 10 virtual IoT clients (~14,027 samples/client); 35,069 held-out global test samples.
- **Feature Dimension:** 192 processed features (one-hot categorical encoding + standard numerical scaling).
- **Network Architecture:** Multilayer Perceptron (192 $\to$ 128 $\to$ 64 $\to$ 32 $\to$ 1 logit).
- **Optimizer & Loss:** Adam ($lr = 0.001$), `BCEWithLogitsLoss`.
- **Rounds & Epochs:** 5 federated rounds, 1 local epoch per round, batch size 256.
- **Client Selection:** Top 7 clients selected per round.
- **Seed:** 42.

---

## 2. Experimental Scenarios

| Experiment ID | Algorithm | Conditions | Purpose |
|---|---|---|---|
| **`exp_a_fedavg_benign`** | FedAvg | Clean, benign clients, no attacks | Baseline federated benchmark |
| **`exp_b_fedavg_attacked`** | FedAvg | Byzantine clients $\{0, 3, 7\}$, sign-flip ($\gamma = 1.0$) | Measure vulnerability of standard FedAvg under adversarial poisoning |
| **`exp_c_trustfed_attacked`** | TrustFed-6G | Byzantine clients $\{0, 3, 7\}$, sign-flip ($\gamma = 1.0$), Dynamic Trust enabled | Evaluate TrustFed-6G defense and adversarial resilience |
| **`exp_d_trustfed_privacy`** | TrustFed-6G | Adaptive Gaussian noise mapped to privacy risk | Evaluate privacy-utility tradeoff |
| **`exp_e_trustfed_clean`** | TrustFed-6G | Clean, benign clients, full TPRS selection | Evaluate multi-criteria resource & quality optimization |

---

## 3. Measured Results Summary

*Measured directly on the held-out UNSW-NB15 global test set (35,069 records):*

| Configuration | Accuracy | Precision | Recall | F1-Score | Duration |
|---|---|---|---|---|---|
| **A: FedAvg (Benign)** | **93.95%** | 92.93% | 98.61% | **95.69%** | 11.72s |
| **B: FedAvg (Attacked)** | **93.21%** | 91.31% | 99.50% | **95.23%** | 9.15s |
| **C: TrustFed-6G (Attacked)** | **93.97%** | 93.13% | 98.40% | **95.69%** | 8.99s |
| **D: TrustFed-6G (Privacy)** | **92.84%** | 90.77% | 99.61% | **94.98%** | 8.55s |
| **E: TrustFed-6G (Clean)** | **93.97%** | 93.13% | 98.40% | **95.69%** | 8.49s |

---

## 4. Key Empirical Findings

1. **Adversarial Resilience:** When malicious clients $\{0, 3, 7\}$ execute a sign-flip update attack under standard FedAvg, global accuracy drops from 93.95% to 93.21% and precision falls to 91.31%. TrustFed-6G successfully detects update directional discrepancy via consensus cosine similarity, degrades the adversaries' trust scores, and restores global accuracy to **93.97%** and F1 to **95.69%**.
2. **Resource & Network Awareness:** TrustFed-6G prioritizes clients with superior compute availability and lower network latency while maintaining optimal convergence.
3. **Privacy-Utility Tradeoff:** Injecting calibrated Gaussian noise based on privacy risk produces a minor utility degradation (accuracy: 92.84%, F1: 94.98%), demonstrating bounded perturbation while concealing raw model updates.
