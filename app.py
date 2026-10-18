"""
TrustFed-6G Interactive Research & Demonstration Dashboard.
Author: TrustFed-6G Research Team
Framework: Streamlit & Plotly
"""

import os
import sys
import json
import glob
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure wide layout and page settings
st.set_page_config(
    page_title="TrustFed-6G Research Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup workspace paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")
CLIENT_DIR = os.path.join(PROJECT_ROOT, "data", "client_splits")
METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")

# Styling: Research Dark Navy Theme
st.markdown("""
<style>
    .main {
        background-color: #0b1120;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-title {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-success { background-color: #064e3b; color: #34d399; }
    .badge-warning { background-color: #78350f; color: #fbbf24; }
    .badge-danger { background-color: #7f1d1d; color: #f87171; }
    .badge-info { background-color: #0c4a6e; color: #38bdf8; }
</style>
""", unsafe_allow_html=True)


def load_all_experiments():
    """Discover all saved experiment runs."""
    runs = {}
    if not os.path.exists(EXPERIMENTS_DIR):
        return runs
    for folder in os.listdir(EXPERIMENTS_DIR):
        m_file = os.path.join(EXPERIMENTS_DIR, folder, "metrics.json")
        if os.path.exists(m_file):
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    runs[folder] = data
            except Exception:
                pass
    return runs


def load_client_metadata():
    path = os.path.join(CLIENT_DIR, "client_metadata.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("TrustFed-6G")
st.sidebar.caption("Adaptive Privacy-Aware Federated Learning for 6G Edge Networks")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Page 1: Overview Dashboard",
        "Page 2: Virtual Edge Clients",
        "Page 3: Federated Training",
        "Page 4: Privacy & Security",
        "Page 5: Results & Comparison",
        "Page 6: About the Project",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Mode:** Live Simulation & Research Engine\n\nNo physical 6G hardware claimed. Software edge simulation on UNSW-NB15 dataset.")

experiments = load_all_experiments()

# ==============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ==============================================================================
if page == "Page 1: Overview Dashboard":
    st.title("🛡️ TrustFed-6G Research Overview")
    st.markdown("Comprehensive surveillance of federated edge intelligence, trust dynamics, and network resilience.")

    if not experiments:
        st.warning("⚠️ No completed experiment runs found yet. You can launch training from **Page 3: Federated Training** or wait for the benchmark suite to finish.")
    else:
        # Select latest or chosen run
        selected_run_id = st.selectbox("Select Experiment Run to View", options=list(experiments.keys()), index=len(experiments)-1)
        run_data = experiments[selected_run_id]
        final_m = run_data.get("final_metrics", {})
        cfg = run_data.get("config", {})

        st.markdown(f"**Experiment ID:** `{selected_run_id}` | **Algorithm:** `{cfg.get('algorithm', 'TrustFed-6G')}` | **Status:** <span class='status-badge badge-success'>Measured Research Run</span>", unsafe_allow_html=True)
        st.markdown("")

        # 4 Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Global Accuracy</div><div class='metric-value'>{final_m.get('accuracy', 0)*100:.2f}%</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>F1-Score</div><div class='metric-value'>{final_m.get('f1', 0)*100:.2f}%</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Precision</div><div class='metric-value'>{final_m.get('precision', 0)*100:.2f}%</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Recall</div><div class='metric-value'>{final_m.get('recall', 0)*100:.2f}%</div></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.subheader("Convergence Across Federated Rounds")
            rounds_hist = run_data.get("round_history", [])
            if rounds_hist:
                df_rounds = pd.DataFrame([
                    {
                        "Round": r["round"],
                        "Accuracy": r["metrics"]["accuracy"],
                        "F1": r["metrics"]["f1"],
                        "Precision": r["metrics"]["precision"],
                        "Recall": r["metrics"]["recall"],
                    }
                    for r in rounds_hist
                ])
                fig = px.line(
                    df_rounds,
                    x="Round",
                    y=["Accuracy", "F1", "Precision", "Recall"],
                    markers=True,
                    template="plotly_dark",
                    title="Evaluation Metrics per Round (Held-Out Global Test Set)",
                    color_discrete_sequence=["#38bdf8", "#34d399", "#f59e0b", "#ec4899"]
                )
                fig.update_layout(yaxis_range=[0.7, 1.0], hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Confusion Matrix")
            cm = final_m.get("confusion_matrix")
            if cm:
                cm_df = pd.DataFrame(cm, index=["Normal (0)", "Attack (1)"], columns=["Pred Normal", "Pred Attack"])
                fig_cm = px.imshow(
                    cm_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    template="plotly_dark",
                    title="Confusion Matrix (Round Final)"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        # Dataset & Pipeline info
        st.subheader("Dataset & Environment Metadata")
        inf_c1, inf_c2, inf_c3, inf_c4 = st.columns(4)
        inf_c1.info("**Dataset:** UNSW-NB15 Network Intrusion")
        inf_c2.info("**Processed Features:** 192 (OneHot + Scaled)")
        inf_c3.info(f"**Total Clients:** {cfg.get('num_clients', 10)} Edge Nodes")
        inf_c4.info(f"**Selected/Round:** {cfg.get('clients_per_round', 7)} Nodes")


# ==============================================================================
# PAGE 2: VIRTUAL EDGE CLIENTS
# ==============================================================================
elif page == "Page 2: Virtual Edge Clients":
    st.title("📱 Virtual Edge Client Profiles & TPRS Metrics")
    st.markdown("Detailed breakdown of simulated 6G edge clients, local resources, network conditions, trust scores, and composite TPRS.")

    if not experiments:
        meta_df = load_client_metadata()
        if meta_df is not None:
            st.dataframe(meta_df, use_container_width=True)
    else:
        selected_run_id = st.selectbox("Inspect Client Data From Run", options=list(experiments.keys()), index=len(experiments)-1)
        run_data = experiments[selected_run_id]
        client_meta = run_data.get("client_metadata_final", [])
        
        if client_meta:
            df_c = pd.DataFrame(client_meta)
            # Reorder columns
            desired_order = ["client_id", "num_samples", "trust", "loss", "cpu_available", "memory_available", "bandwidth", "latency_score", "privacy_risk", "tprs", "selected"]
            cols = [c for c in desired_order if c in df_c.columns] + [c for c in df_c.columns if c not in desired_order]
            df_c = df_c[cols]

            st.dataframe(
                df_c.style.format({
                    "trust": "{:.3f}",
                    "loss": "{:.4f}",
                    "cpu_available": "{:.2f}",
                    "memory_available": "{:.2f}",
                    "bandwidth": "{:.2f}",
                    "latency_score": "{:.2f}",
                    "privacy_risk": "{:.2f}",
                    "tprs": "{:.3f}",
                }).highlight_max(subset=["tprs"], color="#064e3b"),
                use_container_width=True,
            )

            st.markdown("---")
            st.subheader("Client TPRS Component Breakdown")
            c_select = st.selectbox("Select Client for Radar/Bar Inspection", options=df_c["client_id"].tolist())
            c_row = df_c[df_c["client_id"] == c_select].iloc[0]

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                categories = ["Trust", "Model Quality", "Resource", "Network", "Privacy Suit."]
                values = [
                    c_row.get("trust", 0.5),
                    1.0 / (1.0 + float(c_row.get("loss", 1.0))),
                    0.5 * c_row.get("cpu_available", 0.5) + 0.5 * c_row.get("memory_available", 0.5),
                    0.5 * c_row.get("bandwidth", 0.5) + 0.5 * c_row.get("latency_score", 0.5),
                    1.0 - c_row.get("privacy_risk", 0.5)
                ]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=f"Client {c_select}"))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    template="plotly_dark",
                    title=f"Multi-Dimensional Score Profile: Client {c_select}"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with r_col2:
                # Comparison across all clients
                fig_bar = px.bar(
                    df_c,
                    x="client_id",
                    y="tprs",
                    color="selected",
                    color_discrete_map={True: "#38bdf8", False: "#64748b"},
                    title="TPRS Scores & Selection Status Across All Clients",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# PAGE 3: FEDERATED TRAINING
# ==============================================================================
elif page == "Page 3: Federated Training":
    st.title("⚡ Live Federated Training Control Panel")
    st.markdown("Configure and execute real-time federated experiments on the UNSW-NB15 dataset.")

    with st.form("train_config_form"):
        col1, col2 = st.columns(2)
        with col1:
            algorithm = st.selectbox("Aggregation & Selection Algorithm", ["TrustFed-6G", "FedAvg"])
            num_rounds = st.slider("Federated Rounds", min_value=1, max_value=10, value=3)
            clients_per_round = st.slider("Clients Selected Per Round", min_value=2, max_value=10, value=7)
            local_epochs = st.slider("Local Epochs per Round", min_value=1, max_value=5, value=1)
        with col2:
            batch_size = st.selectbox("Batch Size", [64, 128, 256], index=2)
            lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
            seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)
            enable_privacy = st.checkbox("Enable Adaptive Privacy Perturbation", value=False)
            enable_dynamic_trust = st.checkbox("Enable Dynamic Trust Updating (Defense)", value=True)

        exp_name_input = st.text_input("Experiment ID", value=f"live_run_{int(time.time())}")
        submitted = st.form_submit_button("🚀 Start Federated Training")

    if submitted:
        from server.experiment_runner import run_experiment
        cfg = {
            "name": exp_name_input,
            "algorithm": algorithm,
            "rounds": num_rounds,
            "num_clients": 10,
            "clients_per_round": clients_per_round,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "seed": seed,
            "enable_privacy": enable_privacy,
            "enable_dynamic_trust": enable_dynamic_trust,
        }

        status_container = st.empty()
        progress_bar = st.progress(0)
        chart_container = st.empty()

        st.info(f"Training started for `{exp_name_input}`. Tracking per-round global test metrics...")

        def on_round_update(curr_r, tot_r, metrics):
            progress_bar.progress(curr_r / tot_r)
            status_container.markdown(
                f"**Round {curr_r}/{tot_r} Complete** | "
                f"Accuracy: `{metrics['accuracy']*100:.2f}%` | "
                f"F1: `{metrics['f1']*100:.2f}%`"
            )

        try:
            res = run_experiment(cfg, progress_callback=on_round_update)
            st.success(f"🎉 Experiment `{exp_name_input}` completed successfully in {res['duration_seconds']}s!")
            st.json(res["final_metrics"])
        except Exception as e:
            st.error(f"Training failed: {str(e)}")


# ==============================================================================
# PAGE 4: PRIVACY AND SECURITY
# ==============================================================================
elif page == "Page 4: Privacy & Security":
    st.title("🔒 Privacy Preservation & Adversarial Robustness")
    st.markdown("Experimental evaluation of adaptive differential privacy perturbation and byzantine poisoning resistance.")

    st.subheader("1. Adaptive Privacy Perturbation Mechanism")
    st.markdown(r"""
    - **Privacy Unit:** Per-client update vector ($\Delta \theta_i = \theta_{\text{local}, i} - \theta_{\text{global}}$).
    - **L2 Sensitivity Bound:** Vector clipping with threshold $C = 1.0$.
    - **Adaptive Calibration:**
        - $\text{Risk} \ge 0.70 \implies \epsilon = 0.5$ (High protection)
        - $0.40 \le \text{Risk} < 0.70 \implies \epsilon = 1.0$ (Moderate protection)
        - $\text{Risk} < 0.40 \implies \epsilon = 2.0$ (Low protection)
    """)
    st.warning("⚠️ **Research Integrity Notice:** This represents an experimental client-level perturbation simulation. It demonstrates the privacy-utility tradeoff and does not constitute formal cryptographic differential privacy.")

    st.markdown("---")
    st.subheader("2. Controlled Poisoning Attack & TrustFed-6G Defense")
    st.markdown(r"""
    - **Attacking Clients:** IDs `{0, 3, 7}`
    - **Attack Strategy:** Sign-Flip parameter poisoning ($\Delta_{\text{adv}} = -1.0 \times \Delta_{\text{clean}}$)
    - **Defense Mechanism:** Consensus cosine anomaly detection + dynamic trust penalty + TPRS aggregation weighting.
    """)

    # Check if experiments B and C exist
    exp_b = experiments.get("exp_b_fedavg_attacked")
    exp_c = experiments.get("exp_c_trustfed_attacked")
    exp_a = experiments.get("exp_a_fedavg_benign")

    if exp_a and exp_b and exp_c:
        comp_df = pd.DataFrame([
            {"Scenario": "A: Benign FedAvg", "Accuracy": exp_a["final_metrics"]["accuracy"], "F1": exp_a["final_metrics"]["f1"]},
            {"Scenario": "B: Attacked FedAvg (No Defense)", "Accuracy": exp_b["final_metrics"]["accuracy"], "F1": exp_b["final_metrics"]["f1"]},
            {"Scenario": "C: TrustFed-6G Defense", "Accuracy": exp_c["final_metrics"]["accuracy"], "F1": exp_c["final_metrics"]["f1"]},
        ])
        fig_sec = px.bar(
            comp_df,
            x="Scenario",
            y=["Accuracy", "F1"],
            barmode="group",
            template="plotly_dark",
            title="Attack Impact vs TrustFed-6G Defense Resilience",
            color_discrete_sequence=["#38bdf8", "#34d399"]
        )
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        st.info("Run the benchmark suite or individual security experiments to visualize the live attack comparison.")


# ==============================================================================
# PAGE 5: RESULTS AND COMPARISON
# ==============================================================================
elif page == "Page 5: Results & Comparison":
    st.title("📊 Research Benchmark Results & Comparative Study")
    st.markdown("Compare accuracy, precision, recall, and F1-score across all evaluated configurations.")

    if not experiments:
        st.info("No saved experiments found.")
    else:
        summary_rows = []
        for name, data in experiments.items():
            m = data.get("final_metrics", {})
            cfg = data.get("config", {})
            summary_rows.append({
                "Run ID": name,
                "Algorithm": cfg.get("algorithm", "Unknown"),
                "Rounds": cfg.get("rounds", 5),
                "Accuracy (%)": round(m.get("accuracy", 0) * 100, 2),
                "Precision (%)": round(m.get("precision", 0) * 100, 2),
                "Recall (%)": round(m.get("recall", 0) * 100, 2),
                "F1-Score (%)": round(m.get("f1", 0) * 100, 2),
                "Duration (s)": data.get("duration_seconds", 0),
            })
        df_sum = pd.DataFrame(summary_rows)
        st.dataframe(df_sum.style.highlight_max(subset=["Accuracy (%)", "F1-Score (%)"], color="#064e3b"), use_container_width=True)

        st.subheader("Multi-Model Comparison Chart")
        fig_comp = px.bar(
            df_sum,
            x="Run ID",
            y=["Accuracy (%)", "F1-Score (%)"],
            barmode="group",
            template="plotly_dark",
            title="Benchmark Performance Comparison",
            color_discrete_sequence=["#38bdf8", "#34d399"]
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Export CSV
        csv_data = df_sum.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Benchmark Results as CSV",
            data=csv_data,
            file_name="trustfed_benchmark_results.csv",
            mime="text/csv",
        )


# ==============================================================================
# PAGE 6: ABOUT THE PROJECT
# ==============================================================================
elif page == "Page 6: About the Project":
    st.title("📖 About TrustFed-6G")
    st.markdown("""
    ### Project Title
    **TrustFed-6G: Adaptive Privacy-Aware Federated Learning for Secure and Resource-Efficient 6G Edge Networks**

    ---

    ### Abstract & Research Objectives
    The rollout of 6G edge networks promises ultra-low latency, ubiquitous connectivity, and decentralized intelligence. However, training machine learning models over distributed edge devices exposes the network to heterogeneous resource constraints, privacy leakage, and byzantine poisoning attacks.

    **TrustFed-6G** introduces an adaptive, multi-criteria client-selection and aggregation framework:
    1. **TPRS (Trust-Privacy-Resource Score):** Combines historical trust ($T_i$), local model quality ($Q_i$), resource suitability ($R_i$), network suitability ($C_i$), and privacy suitability ($S_i$).
    2. **Trust-Aware Weighted Aggregation:** Computes normalized model aggregation weights proportional to sample volume and TPRS score.
    3. **Adaptive Privacy Perturbation:** Dynamically scales update clipping and Gaussian perturbation based on individual client privacy-risk tiers.
    4. **Dynamic Consensus Defense:** Detects sign-flip adversarial updates through consensus cosine similarity and degrades compromised client trust.

    ---

    ### System Architecture & Mathematical Formulation
    $$\\text{TPRS}_i = 0.30 T_i + 0.25 Q_i + 0.20 R_i + 0.15 C_i + 0.10 S_i$$
    $$\\alpha_i = \\frac{\\text{TPRS}_i \\cdot n_i}{\\sum_{j \\in \\mathcal{S}} \\text{TPRS}_j \\cdot n_j}$$

    ---

    ### Simulation Disclaimer
    *This software prototype runs on simulated edge client splits derived from the UNSW-NB15 benchmark dataset. It does not deploy real physical 6G base stations or physical IoT hardware.*
    """)
