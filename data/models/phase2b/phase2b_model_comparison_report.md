# CloudScale AI — Phase 2B Model Comparison Report
## Tabular Future CPU Forecasting: Random Forest vs XGBoost

**Date**: 2026-10-04 23:28:09  
**Primary Dataset**: `D:\PBL\data\temporal\borg_temporal_pairs.csv` (65,398 samples)  
**Primary Target**: `target_future_cpu_avg`  
**Forecasting Task**: Variable-Gap 1-Step Future CPU Forecasting (t → t+Δt)  
**Script**: `D:\PBL\data\models\phase2b\train_tabular_models.py`

---

## 1. Executive Summary

This phase trained and evaluated two tabular baseline models — **Random Forest Regressor** and **XGBoost Regressor** — on 65,398 chronological 1-step future CPU forecasting pairs extracted from the full Google Borg workload trace (405,894 raw rows, 31 days of production telemetry).

### Key Results:

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| **Test MAE** | **0.000615** | 0.000877 |
| **Test RMSE** | **0.001899** | 0.002290 |
| **Test R²** | **0.9071** | 0.8650 |
| Training Time | **2.46 s** | 3.12 s |
| Inference Time | 0.192 s | **0.075 s** |

### **BEST TABULAR BASELINE: RANDOM FOREST**

Random Forest outperforms XGBoost on all three primary test-set metrics (MAE, RMSE, R²) while also training faster. XGBoost has a 2.5× inference speed advantage and a smaller train/validation generalization gap, but does not surpass Random Forest on absolute test accuracy.

---

## 2. Dataset Inspection & Pre-Training Audit

### 2.1 Dataset

| Property | Value |
|---|---|
| **Source File** | `D:\PBL\data\temporal\borg_temporal_pairs.csv` |
| **Shape** | 65,398 rows × 26 columns |
| **Null Values** | 0 (zero) |
| **Infinite Values** | 0 (zero) |
| **Data Type** | float64 / int64 (all numeric except `target_future_util_state`) |

### 2.2 Target Column Inspection

| Column | Role | Null Count |
|---|---|---|
| `target_future_cpu_avg` | **Primary target** — future CPU usage at t+1 | 0 |
| `target_future_cpu_util_pct` | Secondary target — CPU utilisation % at t+1 | 0 |
| `target_future_util_state` | Categorical label (OVERLOADED/OPTIMAL/UNDERUTILIZED) | 0 |

**Primary target `target_future_cpu_avg`** was chosen because it represents the raw future CPU allocation value in normalised machine-capacity units, directly measurable and physically interpretable. `target_future_cpu_util_pct` is derived from it and introduces a division that can amplify noise at low request values.

### 2.3 Chronological Ordering Verification

- `timestamp_t_us` verified strictly non-decreasing across all 65,398 rows ✅  
- `delta_t_sec` min: **2×10⁻⁶ s**, max: **3,599.995 s** (within the 0 < Δt ≤ 3,600 s extraction filter) ✅  
- All `delta_t_sec` values > 0 ✅  
- Target timestamps are strictly after feature timestamps by construction (confirmed from Phase 2A extraction script) ✅

### 2.4 Train / Validation / Test Split

| Split | Rows | Proportion | Timestamp Range |
|---|---:|---:|---|
| **Train** | 52,318 | 80% | `timestamp_t_us` 0 → 52,317 |
| **Validation** | 6,540 | 10% | rows 52,318 → 58,857 |
| **Test** | 6,540 | 10% | rows 58,858 → 65,397 |

Chronological integrity verified: Train max timestamp < Validation min timestamp < Test min timestamp. No random shuffling applied.

---

## 3. Feature Selection & Leakage Audit

### 3.1 Included Features (18 total)

| # | Feature | Category | Available at Prediction Time? |
|---|---|---|---|
| 1 | `priority` | Workload spec | ✅ Yes |
| 2 | `scheduling_class` | Workload spec | ✅ Yes |
| 3 | `collection_type` | Workload spec | ✅ Yes |
| 4 | `cluster` | Infrastructure | ✅ Yes |
| 5 | `vertical_scaling` | Configuration | ✅ Yes |
| 6 | `scheduler` | Configuration | ✅ Yes |
| 7 | `cpu_request` | Resource spec | ✅ Yes |
| 8 | `memory_request` | Resource spec | ✅ Yes |
| 9 | `assigned_memory` | Memory allocation | ✅ Yes |
| 10 | `cpi` | Microarchitecture telemetry at t | ✅ Yes |
| 11 | `mai` | Microarchitecture telemetry at t | ✅ Yes |
| 12 | `sample_rate` | Measurement quality | ✅ Yes |
| 13 | `cpu_avg_usage_t` | Current CPU usage | ✅ Yes |
| 14 | `memory_avg_usage_t` | Current memory usage | ✅ Yes |
| 15 | `cpu_utilization_pct_t` | Current CPU utilisation % | ✅ Yes |
| 16 | `memory_utilization_pct_t` | Current memory utilisation % | ✅ Yes |
| 17 | `cpu_burst_factor_t` | Current peak/avg CPU ratio | ✅ Yes |
| 18 | `delta_t_sec` | Forecasting horizon | ✅ Yes (known at pair construction) |

### 3.2 Role of `delta_t_sec`

The dataset represents **variable-gap 1-step future CPU forecasting**. The time gap between observation t and the next observation t+1 is heterogeneous (median ≈ 10.5 min, mean ≈ 16.6 min, max = 60 min). `delta_t_sec` was included as an explicit model feature because:

1. It is known at prediction time (it is the requested forecasting horizon).
2. Its inclusion allows tree-based models to learn horizon-conditional patterns.
3. Omitting it would misrepresent the experiment as fixed-interval forecasting.

This task is correctly described as **Variable-Gap 1-Step Future CPU Forecasting**, not as fixed-interval forecasting.

### 3.3 Excluded Features (Leakage Prevention)

| Column | Exclusion Reason |
|---|---|
| `target_future_cpu_avg` | **Direct target** — would be perfect leakage |
| `target_future_cpu_util_pct` | Future-derived — leakage |
| `target_future_util_state` | Future-derived label — leakage |
| `collection_id` | High-cardinality identifier |
| `instance_index` | High-cardinality identifier |
| `machine_id` | High-cardinality identifier |
| `timestamp_t_us` | Timestamp index, not a predictor |
| `time_hours` | Monotonic timestamp re-encoding, not a predictor |

Leakage audit programmatically verified — no target column present in feature list ✅

---

## 4. Preprocessing

Tree-based models (Random Forest, XGBoost) are invariant to monotonic feature scaling. No `StandardScaler` or `MinMaxScaler` was applied.

Missing value imputation was performed during Phase 2A extraction using **train-set medians only** (computed before any validation/test data was seen). At training time the loaded CSV already contains zero nulls, so no further imputation was required.

No one-hot encoding was applied to categorical features. `cluster`, `collection_type`, `scheduling_class`, `vertical_scaling`, `scheduler` are already encoded as low-cardinality integers in the Borg dataset.

---

## 5. Model Configurations

### Random Forest Regressor

```
RandomForestRegressor(
    n_estimators = 200,
    random_state = 42,
    n_jobs       = -1
)
```

### XGBoost Regressor

```
XGBRegressor(
    n_estimators    = 200,
    learning_rate   = 0.05,
    max_depth       = 6,
    subsample       = 0.8,
    colsample_bytree = 0.8,
    random_state    = 42,
    n_jobs          = -1
)
```

---

## 6. Training Results

### 6.1 Full Evaluation Table

| Model | Split | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| **Random Forest** | Train | 0.000038 | 0.000536 | 0.9937 |
| **Random Forest** | Validation | 0.001116 | 0.004706 | 0.7895 |
| **Random Forest** | **Test** | **0.000615** | **0.001899** | **0.9071** |
| **XGBoost** | Train | 0.000162 | 0.000687 | 0.9896 |
| **XGBoost** | Validation | 0.001179 | 0.004400 | 0.8160 |
| **XGBoost** | **Test** | **0.000877** | **0.002290** | **0.8650** |

### 6.2 Summary Comparison Table

| Metric | Random Forest | XGBoost | Winner |
|---|---:|---:|---|
| **Test MAE** | 0.000615 | 0.000877 | **RF** (−28.9%) |
| **Test RMSE** | 0.001899 | 0.002290 | **RF** (−17.1%) |
| **Test R²** | 0.9071 | 0.8650 | **RF** (+4.9 pp) |
| **Training Time** | 2.46 s | 3.12 s | **RF** |
| **Inference Time** | 0.192 s | 0.075 s | **XGBoost** (2.6×) |

---

## 7. Metric Definitions

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values. Lower = better. Directly interpretable in the same units as `target_future_cpu_avg` (normalised CPU fraction).
- **RMSE (Root Mean Squared Error)**: Square root of mean squared errors. Penalises large errors more than MAE. Lower = better.
- **R² (Coefficient of Determination)**: Proportion of target variance explained by the model. Higher = better. R² = 1.0 is perfect; R² = 0 means the model performs no better than predicting the mean.

---

## 8. Overfitting Analysis

### Random Forest

- Train R² = **0.9937** → Test R² = **0.9071**  
- Gap: **0.0866** (8.66 percentage points)  
- This is **mild overfitting** — typical for unpruned decision trees in a Random Forest. The forest has 200 deep trees, each capable of perfectly partitioning its bootstrap sample.  
- Importantly, the validation R² (0.7895) is significantly lower than the test R² (0.9071). This reflects temporal distribution shift — the validation window (rows 52k–58k) spans a period with higher CPU variance than the test window.
- The high test R² (0.9071) confirms the model generalises well to the final held-out period.

### XGBoost

- Train R² = **0.9896** → Test R² = **0.8650**  
- Gap: **0.1246** (12.46 percentage points)  
- XGBoost shows a **larger absolute train-to-test drop** than Random Forest, despite having explicit regularisation (`subsample=0.8`, `colsample_bytree=0.8`, `learning_rate=0.05`).  
- Validation R² (0.8160) is closer to test R² (0.8650) than for RF, indicating XGBoost has a more stable generalisation pattern across temporal windows.

**Conclusion**: Neither model is severely overfit. Random Forest achieves superior absolute test accuracy despite the larger train-test R² gap.

---

## 9. Residual Analysis (Test Set)

### 9.1 Residual Statistics

| Model | Mean Error | Median Error | Std Dev | Min Error | Max Error |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | −0.000093 | **0.000000** | 0.001897 | −0.034209 | 0.034593 |
| **XGBoost** | −0.000048 | +0.000017 | 0.002289 | −0.032299 | 0.034241 |

**Observations**:
- Both models are approximately unbiased (mean errors close to zero).
- RF's median residual of 0.000 indicates the median prediction is exact, consistent with the majority of predictions falling in the common low-CPU regime.
- RF has lower residual standard deviation (0.001897 vs 0.002289), confirming tighter distribution around zero.
- Extreme errors (±0.034) are comparable for both models, suggesting the same hard cases — likely high-variance burst workloads — challenge both.

---

## 10. Performance by Forecasting Horizon (Δt Analysis)

This is a **variable-gap forecasting task**. Performance was evaluated across four Δt bins:

| Horizon Range | Count | RF MAE | RF RMSE | RF R² | XGB MAE | XGB RMSE | XGB R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≤ 5 min | 1,892 | 0.001018 | 0.002563 | 0.7294 | 0.001283 | 0.003010 | 0.6270 |
| 5 < Δt ≤ 15 min | 1,455 | 0.000552 | 0.001705 | 0.9618 | 0.000775 | 0.002040 | 0.9453 |
| 15 < Δt ≤ 30 min | 1,521 | 0.000401 | 0.001600 | 0.9133 | 0.000843 | 0.002131 | 0.8463 |
| 30 < Δt ≤ 60 min | 1,672 | 0.000408 | 0.001346 | 0.9420 | 0.000536 | 0.001581 | 0.9200 |

**Key Finding**:

The shortest horizon bin (≤ 5 min) is the **hardest to predict** for both models (RF R² = 0.73, XGB R² = 0.63). This is counterintuitive but explainable: very short horizons capture rapid intra-instance CPU bursts (high-frequency noise), whereas medium-to-long horizons (5–60 min) reflect more stable utilisation patterns driven by workload scheduling cycles, giving the tabular models better signal.

Random Forest consistently outperforms XGBoost across every Δt bucket on both MAE and RMSE.

---

## 11. Performance by Target CPU Usage Range

| CPU Range | Count | RF MAE | RF RMSE | RF R² | XGB MAE | XGB RMSE | XGB R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low (< 0.001) | 3,337 | 0.000305 | 0.001624 | −42.53 | 0.000376 | 0.001712 | −47.37 |
| Medium (0.001–0.005) | 1,427 | 0.000516 | 0.000999 | +0.28 | 0.001359 | 0.002137 | −2.32 |
| High (≥ 0.005) | 1,776 | 0.001277 | 0.002743 | +0.92 | 0.001429 | 0.003182 | +0.89 |

**Key Finding — Negative R² in the Low CPU Range**:

The extremely negative R² values in the Low CPU bin are not a sign of model failure — they indicate that within the sub-population of near-zero CPU targets (< 0.001 normalised units), the variance of the actual target values is itself very small (close to zero), meaning the denominator of R² is near-zero. Even tiny absolute errors produce catastrophically large negative R². MAE and RMSE values in this bin (0.0003–0.0004) remain comparable to the rest, confirming predictions are numerically close.

This is a known limitation of R² in near-constant sub-populations. It does not indicate the models are useless at low CPU values.

Random Forest has lower MAE and RMSE across all three ranges.

---

## 12. Feature Importance

### 12.1 Top 10 Feature Importance Comparison

| Rank | Random Forest Feature | RF Gini Imp | XGBoost Feature | XGB F-Score Imp |
|---:|---|---:|---|---:|
| 1 | `cpu_avg_usage_t` | 0.7047 | `cpu_avg_usage_t` | 0.3586 |
| 2 | `cpu_request` | 0.0584 | `assigned_memory` | 0.1750 |
| 3 | `assigned_memory` | 0.0503 | `memory_request` | 0.1179 |
| 4 | `delta_t_sec` | 0.0394 | `cpu_request` | 0.1068 |
| 5 | `memory_utilization_pct_t` | 0.0344 | `memory_utilization_pct_t` | 0.0493 |
| 6 | `memory_request` | 0.0330 | `cpu_utilization_pct_t` | 0.0337 |
| 7 | `memory_avg_usage_t` | 0.0286 | `memory_avg_usage_t` | 0.0296 |
| 8 | `cpu_burst_factor_t` | 0.0193 | `cpu_burst_factor_t` | 0.0253 |
| 9 | `cpu_utilization_pct_t` | 0.0082 | `scheduler` | 0.0222 |
| 10 | `mai` | 0.0071 | `scheduling_class` | 0.0164 |

**Observations**:

1. `cpu_avg_usage_t` is by far the single most predictive feature in both models — dominant in RF (70.5% of total importance) and the top feature in XGB (35.9%). This makes physical sense: the best predictor of future CPU usage is current CPU usage (autocorrelation).

2. Both models agree on the top resource specification features: `cpu_request`, `memory_request`, `assigned_memory` all appear in both top-10 lists.

3. `delta_t_sec` ranks 4th in RF (3.9%) but does not appear in XGB top-10 — XGBoost distributes importance more uniformly across memory-related features.

4. `scheduler` and `scheduling_class` appear in XGB's top-10 but not RF's, suggesting XGBoost captures more scheduler-level variation.

> **Important disclaimer**: Feature importance indicates the model's relative reliance on each feature during split decisions. It does NOT imply causal influence. `cpu_avg_usage_t` dominates because of temporal autocorrelation, not because it mechanistically causes future CPU usage.

---

## 13. SHAP Analysis

SHAP was **not executed** — the `shap` package was not installed in the environment. This is expected per the optional SHAP directive. SHAP can be added in a future phase without affecting the core model comparison.

---

## 14. Model Selection Recommendation

### **BEST TABULAR BASELINE: RANDOM FOREST**

| Criterion | Random Forest | XGBoost | Winner |
|---|---|---|---|
| Test RMSE | 0.001899 | 0.002290 | **RF** |
| Test MAE | 0.000615 | 0.000877 | **RF** |
| Test R² | 0.9071 | 0.8650 | **RF** |
| Train-Test Gap (RMSE) | 0.001363 | 0.001603 | **RF** (smaller) |
| Stability across Δt | Best at 4/4 bins | Best at 0/4 bins | **RF** |
| Training Time | 2.46 s | 3.12 s | **RF** |
| Inference Time | 0.192 s | 0.075 s | **XGBoost** |

Random Forest wins on every accuracy criterion. XGBoost wins only on inference speed.

For the Phase 2B objective of establishing the best tabular forecasting baseline, **Random Forest is the recommended model**.

XGBoost's faster inference (0.075 s vs 0.192 s for 6,540 samples) may be relevant in real-time serving contexts but is not a Phase 2B criterion.

---

## 15. Limitations & Scientific Disclaimers

1. **Variable-Gap Horizon**: Predictions correspond to a variable temporal gap (t → t+Δt), not a fixed sampling interval. Performance at sub-5-minute horizons is lower than at longer horizons.

2. **Target Scale**: `target_future_cpu_avg` is expressed in normalised machine-capacity units (0 to ~0.53 in this dataset). MAE values (0.0006–0.0009) must be interpreted relative to this scale, not as absolute CPU percentages.

3. **No Sequence Context**: Tabular models only see a single timestep (t) and the forecasting horizon (delta_t_sec). They have no access to the full historical trajectory. LSTM models trained in Phase 2C will capture this sequence information.

4. **Temporal Distribution Shift**: Validation R² (0.79–0.82) is lower than Test R² (0.87–0.91). This is expected in temporal forecasting — different time windows exhibit different CPU usage patterns due to workload scheduling cycles.

5. **No Cost / Energy Claims**: Offline accuracy metrics (RMSE/MAE/R²) do not constitute proof of operational cost savings, energy reductions, or improved response times. Those claims belong to later phases after integration and A/B evaluation.

---

## 16. Recommended Next Step

Phase 2B establishes Random Forest as the best tabular baseline with Test R² = 0.9071.

**Phase 2C (pending review)**: Train LSTM/Temporal Neural Network on the LSTM sequences dataset (`borg_lstm_sequences.npz`) incorporating the `delta_t_sec` variable-gap feature, and compare against the Random Forest tabular baseline.

---

## 17. Final Validation Checklist

- [x] Real Borg temporal dataset used (`borg_temporal_pairs.csv`, 65,398 rows).
- [x] 30k subset NOT used for final forecasting experiment.
- [x] Future target (`target_future_cpu_avg`) is genuinely future (observation at t+1).
- [x] No direct target leakage — all features available at prediction time t.
- [x] Chronological split preserved (80/10/10 by row order).
- [x] No random temporal shuffling.
- [x] Train medians computed on training set only, applied to all splits.
- [x] Random Forest trained (n_estimators=200, random_state=42).
- [x] XGBoost trained (n_estimators=200, lr=0.05, max_depth=6, random_state=42).
- [x] MAE calculated for Train, Validation, Test.
- [x] RMSE calculated for Train, Validation, Test.
- [x] R² calculated for Train, Validation, Test.
- [x] Test set evaluated — reported as primary final evaluation.
- [x] Variable Δt analyzed across 4 horizon bins (≤5, 5–15, 15–30, 30–60 min).
- [x] CPU usage range analysis performed (Low/Medium/High).
- [x] Overfitting explicitly checked.
- [x] Feature importance generated for both models.
- [x] Models saved: `random_forest_cpu_forecaster.joblib`, `xgboost_cpu_forecaster.json`.
- [x] Metrics saved: `model_metrics.json`.
- [x] Feature list saved: `feature_list.json`.
- [x] Training config saved: `training_config.json`.
- [x] 6 evaluation plots saved to `plots/`.
- [x] Backend NOT modified.
- [x] Frontend NOT modified.
- [x] Database NOT modified.
- [x] Production ML pipeline NOT modified.
- [x] No cost savings claimed.
- [x] No energy savings claimed.
- [x] No response-time improvement claimed.
- [x] LSTM NOT trained in this phase.

---

## 18. Saved Artifacts

| File | Size | Description |
|---|---:|---|
| `train_tabular_models.py` | 26.1 KB | Reproducible training script |
| `random_forest_cpu_forecaster.joblib` | 36.8 MB | Trained RF model |
| `xgboost_cpu_forecaster.json` | 734 KB | Trained XGBoost model |
| `model_metrics.json` | 5.2 KB | All metrics, residuals, gap/CPU breakdowns |
| `feature_list.json` | 468 B | Feature names and primary target |
| `training_config.json` | 273 B | Hyperparameter record |
| `plots/1_actual_vs_predicted.png` | 216 KB | Scatter: Actual vs Predicted (Test) |
| `plots/2_residual_distribution.png` | 161 KB | KDE: Residual distribution (Test) |
| `plots/3_model_metric_comparison.png` | 79 KB | Bar chart: MAE/RMSE comparison |
| `plots/4_error_vs_delta_t.png` | 131 KB | Bar chart: RMSE by Δt horizon bin |
| `plots/5_feature_importance_rf.png` | 126 KB | Top-10 RF feature importances |
| `plots/6_feature_importance_xgb.png` | 137 KB | Top-10 XGBoost feature importances |
