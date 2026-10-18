# TrustFed-6G Website & Dashboard User Guide

The **TrustFed-6G** dashboard is a responsive, research-oriented demonstration web application built with Streamlit and Plotly.

## 1. How to Launch the Web Application

Open Windows PowerShell in the project directory:

```powershell
cd C:\Users\HP\Downloads\TrustFed-AI-main\TrustFed-AI-main
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 2. Navigating the 6 Research Pages

### Page 1: Overview Dashboard
- **Purpose:** High-level executive view of the federated network.
- **Features:**
  - Selector to toggle between completed experiment runs (e.g., `exp_a_fedavg_benign`, `exp_c_trustfed_attacked`).
  - Key metric cards: Global Accuracy, F1-Score, Precision, Recall.
  - Interactive Plotly line chart showing convergence trajectory across rounds.
  - Confusion matrix heatmap evaluated on the 35,069-sample held-out test set.
  - Dataset and model architecture metadata badges.

### Page 2: Virtual Edge Clients
- **Purpose:** Deep inspection of the 10 virtual IoT edge clients.
- **Features:**
  - Interactive table displaying client sample counts, observed loss, trust scores, CPU/RAM telemetry, network latency, privacy risk, and composite TPRS score.
  - Multi-dimensional Radar Chart profiling individual clients across all 5 evaluation axes.
  - Bar chart visualizing TPRS distribution and top-7 selection status.

### Page 3: Federated Training
- **Purpose:** Interactive control center to launch real live training runs.
- **Features:**
  - Configurable parameters: algorithm selection (TrustFed-6G vs. FedAvg), number of rounds, clients per round, local epochs, batch size, learning rate, and random seed.
  - Checkboxes to toggle Adaptive Privacy Perturbation and Dynamic Trust Defense.
  - Live progress bar and per-round metric streaming.

### Page 4: Privacy & Security
- **Purpose:** Evaluating privacy-utility tradeoffs and byzantine attack defense.
- **Features:**
  - Specification of adaptive privacy risk mapping and update clipping.
  - Comparative bar charts contrasting Benign FedAvg, Attacked FedAvg (without defense), and TrustFed-6G (with defense).

### Page 5: Results & Comparison
- **Purpose:** Comparative benchmarking table and data export.
- **Features:**
  - Unified table summarizing all completed experiments.
  - Multi-bar comparison of Accuracy and F1-Score.
  - One-click button to download benchmark results as a CSV spreadsheet.

### Page 6: About the Project
- **Purpose:** Research context for Viva & review presentations.
- **Features:**
  - Abstract, problem statement, mathematical formulas, and simulation disclaimers.
