# CloudScale AI — Phase 4: Model Selection Summary

**Generated Date**: 2026-10-05  
**Project**: Machine Learning-Driven Auto Scaling and Cost Optimization Framework for Cloud Resources  
**Phase**: Phase 4 — Final Model Comparison & Selection  

---

## 1. Selected Model Roles

- **PRIMARY MODEL**: **LSTM**
- **FALLBACK MODEL**: **Random Forest**
- **BENCHMARK MODEL**: **XGBoost**

---

## 2. Executive Rationale & Evidence

1. **Superior Prediction Accuracy**:
   - Achieved the highest coefficient of determination ($R^2 = 0.9838$).
   - On their respective chronological test evaluations, the LSTM achieved a 71.2% lower RMSE than the reported Random Forest baseline. Because the models were evaluated on different-sized evaluation populations, this should not be interpreted as a perfectly controlled sample-level head-to-head comparison.
   - Lowest Mean Absolute Error ($\text{MAE} = 0.000417$), representing a **32.20% MAE reduction** over Random Forest ($\text{MAE} = 0.000615$) and a **52.45% MAE reduction** over XGBoost ($\text{MAE} = 0.000877$).

2. **Sequential Multi-Step Modeling**:
   - Unlike Random Forest and XGBoost, which treat historical observations as independent tabular features in isolated $t \to t+1$ pairs, the LSTM architecture explicitly processes 5 consecutive historical time steps ($t-4 \dots t$). This enables the model to capture non-linear temporal momentum, trend trajectories, and workload phase transitions.

3. **Consistent Error Behavior Across Temporal Gap Bins**:
   - Within the LSTM evaluation subset, R² remained above 0.97 across the three evaluated temporal-gap bins. However, direct bin-level comparison with Random Forest and XGBoost should be interpreted cautiously because the sequence and tabular evaluation populations are not identical.

4. **Scientifically Validated & Leakage-Free**:
   - The independent Phase 3 post-training audit confirmed that the LSTM implementation is strictly leakage-free: `delta_t_sec` was excluded to eliminate future target timestamp leakage, scalers were fitted exclusively on training data, and chronological split boundaries were preserved.

5. **Inference Latency Benchmark**:
   - Measured single-sample LSTM inference latency was approximately 0.237 ms in the Phase 4 benchmark. The appropriate production decision-loop requirement will be established during later system integration.

---

## 3. Important Selected Model Limitations

1. **Not Explicitly Time-Gap-Aware**:
   - `delta_t_sec` was removed to prevent temporal leakage at prediction time. As a result, the baseline LSTM treats input steps as sequentially ordered without explicitly measuring elapsed seconds between non-uniform steps.

2. **Sequence Buffering Requirement**:
   - Requires a sequence buffer of 5 historical observations per workload instance before generating a forecast. Cold-start instances with fewer than 5 observations cannot be forecasted directly by the LSTM.

3. **Higher Deployment Complexity**:
   - Requires bundling Python deep learning runtimes (`tensorflow` / Keras), `feature_scaler.pkl`, `target_scaler.pkl`, and sequence data structures into backend services, whereas tabular models require simpler tree-based execution.

4. **Black-Box Interpretability**:
   - Feature importances for the LSTM were not independently established via SHAP or Integrated Gradients in Phase 3, unlike tree-based feature importance vectors available in RF and XGBoost.

5. **Limited Multi-Step Horizon**:
   - Evaluated exclusively for 1-step-ahead forecasting ($t+1$, max 30 min gap). Long-term multi-step forecasting (> 30 min) requires multi-step auto-regressive unrolling in future phases.

---

## 4. Why Other Models Were Not Selected

### Random Forest (FALLBACK MODEL)
- **Why Fallback**: While RF performed well ($R^2 = 0.9071$, $\text{RMSE} = 0.001899$), its RMSE is higher than the LSTM on their respective test evaluations. RF processes single-step pairs without memory of sequence trajectories, but serves as an ideal fallback handler for cold-start instances with < 5 sequence steps.

### XGBoost (BENCHMARK MODEL)
- **Why Benchmark**: XGBoost recorded $R^2 = 0.8650$ and $\text{RMSE} = 0.002290$. It demonstrated high sensitivity to memory features rather than temporal trends, serving as a tabular baseline reference.

---

## 5. Summary Performance Table

| Model | Model Role | Test MAE | Test RMSE | Test R² | Relative RMSE vs LSTM | Infer Latency (ms/sample) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Fallback Model | 0.000615 | 0.001899 | 0.9071 | +247.2% | 0.0293 ms |
| **XGBoost** | Benchmark Model | 0.000877 | 0.002290 | 0.8650 | +318.9% | 0.0115 ms |
| **LSTM** | Primary Model | **0.000417** | **0.000547** | **0.9838** | **Baseline (Best)** | 0.2369 ms |

---

## 6. Recommended Phase 5 Integration Architecture

- Use **LSTM (`lstm_cpu_forecaster.keras`)** as the **PRIMARY MODEL** for CloudScale AI.
- Maintain **Random Forest** as the **FALLBACK MODEL** for cold-start workload instances that have fewer than 5 historical sequence steps.
- Retain **XGBoost** as a **BENCHMARK MODEL** for reference comparison.
- Expose the primary forecaster cleanly through the backend service pipeline to feed predicted CPU utilization into the auto-scaling and cost optimization decision engine.
