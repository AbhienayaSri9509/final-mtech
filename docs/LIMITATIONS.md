# TrustFed-6G Implementation Limitations & Research Scope

## 1. Simulation Limitations
1. **Software Edge Simulation:** The 10 IoT clients are simulated locally as in-memory PyTorch training loops on disjoint data partitions of the UNSW-NB15 benchmark. Physical 6G radio access networks (RAN), sub-THz propagation channels, and physical IoT embedded hardware were not deployed.
2. **Synchronous Federated Communication:** Rounds are modeled synchronously. Real-world 6G deployments may experience asynchronous client dropouts, stragglers, and packet loss.

## 2. Privacy Limitations
1. **Empirical Noise vs. Formal DP:** The adaptive privacy mechanism clips L2-norm update vectors and applies calibrated Gaussian noise inversely proportional to client privacy risk tiers. However, this does not implement a formal cryptographic privacy accountant (such as Rényi Differential Privacy or Opacus DP-SGD with composition bounds across rounds). Claims of privacy protection must be understood as experimental perturbation defense against parameter inversion.

## 3. Threat Model Assumptions
1. **Consensus Direction Requirement:** The cosine anomaly detection mechanism assumes that the majority of participating clients in a round are non-colluding and benign. If byzantine adversaries constitute $> 50\%$ of selected clients, consensus direction can be subverted.
2. **Evaluated Attacks:** The evaluated attack vectors focused on sign-flip, scaling, and additive Gaussian poisoning. Advanced targeted backdoor attacks and Sybil identity spoofing were not evaluated in this release.
