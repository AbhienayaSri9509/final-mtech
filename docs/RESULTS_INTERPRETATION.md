# TrustFed-6G Results Interpretation & Viva Presentation Guide

## 1. How to Interpret the Metrics

### Accuracy vs. F1-Score in Intrusion Detection
In network intrusion detection (UNSW-NB15), attacks represent abnormal network events. Evaluating both Accuracy and F1-score is essential:
- **Precision:** Measures the fraction of identified attacks that are genuine intrusions (critical to prevent alert fatigue).
- **Recall:** Measures the fraction of actual attacks successfully detected (critical to prevent malicious breach).
- **F1-Score:** Harmonic mean of precision and recall.

### Baseline vs. TrustFed-6G
- In clean settings without attackers, both FedAvg and TrustFed-6G achieve ~93.95% to 93.97% accuracy on UNSW-NB15 after 5 rounds. This confirms that introducing multi-criteria selection (resource, latency, and privacy scores) does not degrade convergence.
- In adversarial environments (sign-flip poisoning), FedAvg suffers an accuracy drop to 93.21% and precision drop to 91.31%. TrustFed-6G neutralizes the attack and maintains 93.97% accuracy.

---

## 2. Viva Defense Q&A Preparation

**Q1: Why did you design the TPRS scoring formula?**  
*A:* Edge devices in 6G networks are heterogeneous. Selecting clients solely on sample size (as in FedAvg) ignores device battery, compute availability, network latency, and malicious behaviour. TPRS provides a unified multi-criteria decision metric balancing trust, data quality, compute, network, and privacy.

**Q2: How does TrustFed-6G defend against sign-flip poisoning?**  
*A:* When malicious clients invert their gradient updates, their update vectors exhibit negative cosine similarity with the consensus direction of the benign clients. Dynamic trust tracking detects this divergence, reduces their trust score $T_i$, lowers their TPRS, and drastically reduces their aggregation weight $\alpha_i$.

**Q3: Does the privacy mechanism offer formal Differential Privacy?**  
*A:* As documented in our implementation audit and protocol, our privacy module simulates an adaptive Gaussian perturbation mechanism with L2 sensitivity clipping. We honestly report this as an empirical privacy perturbation rather than claiming formal differential privacy without a full RDP moments accountant.
