# CloudScale AI — Phase 4: Final Model Comparison and Selection Report

**Generated Date**: 2026-10-05  
**Project**: Machine Learning-Driven Auto Scaling and Cost Optimization Framework for Cloud Resources  
**Phase**: Phase 4 — Final Model Comparison & Selection  

---

## 1. Executive Summary

Phase 4 conducts a rigorous, evidence-based model comparison across the three machine learning models developed during Phase 2B and Phase 3:
1. **Random Forest** (Phase 2B — Tabular ensemble)
2. **XGBoost** (Phase 2B — Gradient boosted decision trees)
3. **LSTM** (Phase 3 — Stacked Long Short-Term Memory neural network)

Based on empirical evidence, model roles are designated as:
- **PRIMARY MODEL**: **LSTM**
- **FALLBACK MODEL**: **Random Forest**
- **BENCHMARK MODEL**: **XGBoost**

On their respective chronological test evaluations, the LSTM achieved a 71.2% lower RMSE than the reported Random Forest baseline. Because the models were evaluated on different-sized evaluation populations, this should not be interpreted as a perfectly controlled sample-level head-to-head comparison. Measured single-sample LSTM inference latency was approximately 0.237 ms in the Phase 4 benchmark. The appropriate production decision-loop requirement will be established during later system integration.

---

## 2. Models Compared

| Model | Model Role | Phase | Architecture / Algorithm | Input Representation | Features Used |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Fallback Model | Phase 2B | Ensemble of 200 Decision Trees | Tabular pairs ($t \to t+1$) | 18 features (inc. `delta_t_sec`) |
| **XGBoost** | Benchmark Model | Phase 2B | Gradient Boosted Trees (lr=0.05, depth=6) | Tabular pairs ($t \to t+1$) | 18 features (inc. `delta_t_sec`) |
| **LSTM** | Primary Model | Phase 3 | Stacked LSTM (64 → 32 units) + Dense(16) | 3D Sequences ($5 \times 7$) | 7 features (excl. `delta_t_sec`) |

---

## 3. Dataset & Evaluation Context

All models were developed using Google Borg workload trace data (`borg_traces_data.csv`, 405,894 rows, 31.01 days).
- **Tabular Models (RF / XGBoost)**: Evaluated on `borg_temporal_pairs.csv` ($65,398$ pairs total, $6,540$ test pairs).
- **LSTM Model**: Evaluated on `borg_lstm_sequences.npz` ($15,594$ 3D sequence tensors total, $1,560$ test sequences).
- **Split Strategy**: Chronological split across 80% Train, 10% Validation, and 10% Test without random shuffling.

---

## 4. Metric Definitions

All models predict `target_future_cpu_avg` (future CPU average usage fraction). Primary metrics evaluated on original unscaled CPU fraction:
- **MAE (Mean Absolute Error)**: $\frac{1}{N}\sum |y_i - \hat{y}_i|$
- **RMSE (Root Mean Squared Error)**: $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$
- **R² (Coefficient of Determination)**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

*Note: R² measures variance explained by the regression model. It is NOT referred to as accuracy.*

---

## 5. Overall Performance

| Model | Model Role | Test MAE | Test RMSE | Test R² | MAE Reduction vs Baseline | RMSE Reduction vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Fallback Model | 0.000615 | 0.001899 | 0.9071 | Ref (Baseline) | Ref (Baseline) |
| **XGBoost** | Benchmark Model | 0.000877 | 0.002290 | 0.8650 | +42.60% (Worse) | +20.59% (Worse) |
| **LSTM** | Primary Model | **0.000417** | **0.000547** | **0.9838** | **-32.20% (Best)** | **-71.20% (Best)** |

On their respective chronological test evaluations, the LSTM achieved a 71.2% lower RMSE than the reported Random Forest baseline. Because the models were evaluated on different-sized evaluation populations, this should not be interpreted as a perfectly controlled sample-level head-to-head comparison.

---

## 6. Temporal Gap Analysis

Evaluation across temporal gap bins ($\Delta t$ between historical state and target):

| Temporal Gap Bin | Random Forest (RMSE / R²) | XGBoost (RMSE / R²) | LSTM (RMSE / R²) | LSTM Sample Count (N) |
| :--- | :--- | :--- | :--- | :--- |
| **≤ 5 min** | 0.002563 / 0.7294 | 0.003010 / 0.6270 | **0.000604 / 0.9882** | 401 |
| **5–15 min** | 0.001705 / 0.9618 | 0.002040 / 0.9453 | **0.000524 / 0.9812** | 1,024 |
| **15–30 min** | 0.001600 / 0.9133 | 0.002131 / 0.8463 | **0.000534 / 0.9734** | 135 |
| **30–60 min** | 0.001346 / 0.9420 | 0.001581 / 0.9200 | — (N=0) | 0 |

Within the LSTM evaluation subset, R² remained above 0.97 across the three evaluated temporal-gap bins. However, direct bin-level comparison with Random Forest and XGBoost should be interpreted cautiously because the sequence and tabular evaluation populations are not identical.

---

## 7. Computational Comparison

| Model | Training Time | Total Test Inference Time | Single-Sample Latency | Parameter Count / Model Size |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 2.46 seconds | 0.192 seconds (6,540 samples) | 0.0293 ms | 200 trees (38.6 MB `.joblib`) |
| **XGBoost** | 3.12 seconds | 0.075 seconds (6,540 samples) | 0.0115 ms | 200 trees (0.75 MB `.json`) |
| **LSTM** | ~35.0 seconds (30 epochs) | 0.370 seconds (1,560 samples) | 0.2369 ms | 31,393 parameters (0.42 MB `.keras`) |

Measured single-sample LSTM inference latency was approximately 0.237 ms in the Phase 4 benchmark. The appropriate production decision-loop requirement will be established during later system integration.

---

## 8. Model Complexity

- **Random Forest / XGBoost**: Tabular input. Low operational complexity. Fast tree evaluation.
- **LSTM**: 3D tensor input ($5 \times 7$). Moderate operational complexity. Requires historical sequence buffering per instance and Python ML runtime dependencies (`tensorflow` / Keras).

---

## 9. Interpretability

- **Random Forest Top Features**: `cpu_avg_usage_t` (70.47%), `cpu_request` (5.84%), `assigned_memory` (5.03%), `delta_t_sec` (3.94%).
- **XGBoost Top Features**: `cpu_avg_usage_t` (35.86%), `assigned_memory` (17.50%), `memory_request` (11.79%), `cpu_request` (10.68%).
- **LSTM Interpretability**: Feature importances were not independently established via SHAP or Integrated Gradients in Phase 3. LSTM functions as a black-box sequential regressor.

---

## 10. Generalization & Chronological Distribution Shift

The Borg trace dataset exhibits chronological distribution shift across 31 days:
- **Train Set (hours 0–504)**: High volatility with peak CPU usage spikes reaching $0.090$.
- **Validation Set (hours 504–569)**: Low activity period with max CPU usage of $0.0040$.
- **Test Set (hours 569–744)**: Moderate activity period with max CPU usage of $0.0298$.

LSTM performance across splits:
- Train $R^2 = 0.8432$ ($\text{RMSE} = 0.001951$)
- Validation $R^2 = 0.9877$ ($\text{RMSE} = 0.000163$)
- Test $R^2 = 0.9838$ ($\text{RMSE} = 0.000547$)

The Phase 3 audit confirmed that lower validation/test error is a legitimate consequence of lower target variance during those chronological windows.

---

## 11. Fairness of Comparison

- **Assessment**: Informative at the dataset level, but not perfectly controlled at the single-sample level.
- **Explanation**: Tabular models were evaluated on $6,540$ test pairs (18 features), whereas LSTM was evaluated on $1,560$ 3D test sequence tensors (7 features). Both datasets derive from the same underlying Google Borg trace source and follow identical 80/10/10 chronological splits, validating the dataset-level comparative findings.

---

## 12. Practical CloudScale AI Suitability

| Criterion | Random Forest | XGBoost | LSTM (Selected) |
| :--- | :--- | :--- | :--- |
| **Prediction Quality** | Good ($R^2 = 0.9071$) | Moderate ($R^2 = 0.8650$) | **Excellent ($R^2 = 0.9838$)** |
| **Temporal Modeling** | Limited (Single step pair) | Limited (Single step pair) | **Excellent (5-step sequence)** |
| **Inference Efficiency** | Excellent ($0.029$ ms) | **Excellent ($0.011$ ms)** | Good ($0.237$ ms) |
| **Interpretability** | **Excellent (SHAP / MDI)** | **Excellent (Gain / Cover)** | Limited (Black-box) |
| **Deployment Simplicity** | **High (Scikit-learn)** | **High (XGBoost JSON)** | Moderate (Keras / TF) |
| **Cold-Start Handling** | **Supported (1 observation)** | **Supported (1 observation)** | Requires 5 sequence steps |

---

## 13. Limitations

1. **Baseline LSTM (Not Explicitly Time-Gap Aware)**: `delta_t_sec` was excluded to prevent temporal leakage; input steps are treated as sequential observations without explicit interval deltas.
2. **Cold-Start Sequence Buffer**: LSTM requires 5 historical sequence observations per workload instance.
3. **No Direct Cost/Energy Claims**: Forecasting accuracy measures CPU usage prediction quality; cost optimization and auto-scaling logic will be implemented in subsequent phases.

---

## 14 & 15. Final Model Selection & Justification

- **PRIMARY MODEL**: **LSTM**
- **FALLBACK MODEL**: **Random Forest**
- **BENCHMARK MODEL**: **XGBoost**

### Justification:
1. On their respective chronological test evaluations, the LSTM achieved a 71.2% lower RMSE than the reported Random Forest baseline. Because the models were evaluated on different-sized evaluation populations, this should not be interpreted as a perfectly controlled sample-level head-to-head comparison.
2. **Sequential Pattern Recognition**: Effectively leverages 5-step historical workload context.
3. Measured single-sample LSTM inference latency was approximately 0.237 ms in the Phase 4 benchmark. The appropriate production decision-loop requirement will be established during later system integration.
4. **Leakage-Free Validation**: Passed strict post-training scientific audit.

---

## 16. Recommended Phase 5 Integration Architecture

- Deploy **LSTM (`lstm_cpu_forecaster.keras`)** as the **PRIMARY MODEL** for CloudScale AI.
- Maintain **Random Forest** as the **FALLBACK MODEL** for new/cold-start workload instances with fewer than 5 historical sequence steps.
- Retain **XGBoost** as a **BENCHMARK MODEL** for reference comparison.
- Pass predicted CPU utilization into the downstream CloudScale AI optimization engine.

---

## 17. Conclusion

Phase 4 successfully evaluated Random Forest, XGBoost, and LSTM models using empirical evidence. The **LSTM model** is selected as the primary forecasting engine for CloudScale AI.

**PHASE 4 STATUS: COMPLETE**
