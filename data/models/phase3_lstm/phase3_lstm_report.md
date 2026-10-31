# CloudScale AI — Phase 3: LSTM Time-Series Forecasting
## Variable-Gap 1-Step-Ahead CPU Utilization Forecasting

**Generated**: 2026-10-05 11:07:45  
**Project**: Machine Learning-Driven Auto Scaling and Cost Optimization Framework  
**Phase**: 3 of 4 — LSTM Model Development (Strict Model Development Only)

---

## 1. Objective

Develop, train, and evaluate an LSTM-based time-series forecasting model for
variable-gap 1-step-ahead CPU utilization prediction using the Google Borg
workload trace dataset.

The model predicts `target_future_cpu_avg` (CPU average usage at t+1) from a
sequence of 5 historical observations. This result will later be compared with
Phase 2B's Random Forest and XGBoost models in Phase 4.

**This phase is model development only. No integration with API, frontend, or
database is performed.**

---

## 2. Dataset Used

| Item | Value |
|------|-------|
| Source NPZ | `D:/PBL/data/temporal/borg_lstm_sequences.npz` |
| Derived from | `borg_traces_data.csv` (405,894 rows, 31.01 days) |
| Workload grouping | `(collection_id, instance_index)` |
| Extraction method | Strict sliding window L=5, max internal gap ≤ 1800 s |

---

## 3. Dataset Dimensions

| Split | Sequences | % |
|-------|-----------|---|
| Train | 12,475 | 80% |
| Validation | 1,559 | 10% |
| Test | 1,560 | 10% |
| **Total** | **15,594** | 100% |

- **Input tensor shape**: X ∈ [12475, 5, 7] (after feature removal)
- **Target shape**: Y ∈ [12475] (scalar per sequence)

---

## 4. Sequence Structure

Each LSTM sequence consists of 5 consecutive historical time-step observations
from the same workload instance, followed by a 1-step-ahead target:

```
[t-4, t-3, t-2, t-1, t]  →  target at t+1
```

Sequences are derived from the same workload instance (`collection_id`,
`instance_index`), ensuring that consecutive steps belong to the same job.

---

## 5. Feature Definitions

Original features in NPZ (8 total):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | priority | Job scheduling priority (0–360) |
| 1 | scheduling_class | Borg scheduling class (0–3) |
| 2 | cpu_request | Requested CPU fraction |
| 3 | memory_request | Requested memory fraction |
| 4 | assigned_memory | Assigned memory fraction |
| 5 | cpu_avg_usage_t | CPU average usage at current step |
| 6 | cpu_utilization_pct_t | CPU utilization percentage at current step |
| 7 | delta_t_sec | **REMOVED** — See Section 9 (leakage audit) |

**Features used for LSTM training: 7 (indices 0–6)**

---

## 6. Target Definition

| Item | Value |
|------|-------|
| Target name | `target_future_cpu_avg` |
| Definition | CPU average usage of the NEXT observation (t+1) |
| Units | Raw CPU fraction (0.0 – ~0.09 in training data) |
| Scaling applied | MinMaxScaler fitted on training data only |

No target-derived feature is included in the input.

---

## 7. Temporal Forecasting Formulation

This is a **variable-gap 1-step-ahead CPU forecasting** task:

- Time gaps between consecutive observations are NOT uniform
- Internal sequence gaps range: 0 – 1800 seconds (max 30 minutes)
- Median internal gap (training): ~279 seconds (~4.7 minutes)
- Mean internal gap (training): ~402 seconds (~6.7 minutes)
- The model predicts the NEXT observation, not a fixed-interval future point

This is explicitly **NOT** fixed-interval time-series forecasting.

---

## 8. Variable-Gap Handling

The dataset has non-uniform temporal sampling (97.83% of sequences have
irregular inter-step spacing). Strategies applied:

- `delta_t_sec` was originally included as feature #8 to encode temporal gaps
- After the mandatory leakage audit (Section 9), it was removed
- The LSTM learns from the 7 remaining features without explicit gap encoding
- This is a documented limitation (see Section 21)

---

## 9. delta_t_sec Leakage Audit

**Status: TEMPORAL LEAKAGE CONFIRMED — FEATURE REMOVED**

### Analysis

From the preprocessing code (`extract_temporal_dataset.py`, lines 265–286),
for a window of 6 observations [w0, w1, w2, w3, w4, w5]:

```
step 0 (t-4): features of w0, delta_t = time(w1) − time(w0)
step 1 (t-3): features of w1, delta_t = time(w2) − time(w1)
step 2 (t-2): features of w2, delta_t = time(w3) − time(w2)
step 3 (t-1): features of w3, delta_t = time(w4) − time(w3)
step 4 (t)  : features of w4, delta_t = time(w5) − time(w4)  ← LEAKAGE
target       : cpu_avg_usage from w5
```

At step 4 (most recent historical observation), `delta_t_sec` encodes the
time interval between the current observation and the **future target
observation** (w5). At actual prediction time, w5 has not yet occurred, so
its timestamp is unknown.

### Decision

**`delta_t_sec` is REMOVED from LSTM input features.**

- Steps 0–3 deltas are historical and technically available at prediction time
- Step 4 delta is future information → leakage
- Because the NPZ cannot be modified (strict rule), and because correcting
  only step 4 would require per-sample manipulation, the conservative safe
  choice is to remove the feature entirely
- LSTM trains with 7 features (indices 0–6)

---

## 10. Data Validation

| Check | Result |
|-------|--------|
| NaN values in X_train | 0 |
| NaN values in Y_train | 0 |
| Inf values anywhere | 0 |
| delta_t_sec > 1800 s in X_train | 0 |
| Target values in [0, 1] | ✓ (max = 0.0900) |
| Sequences with negative delta | 0 |

All 15,594 sequences passed data integrity checks.

---

## 11. Train / Validation / Test Split

| Split | Sequences | Method |
|-------|-----------|--------|
| Train | 12,475 | First 80% chronologically |
| Validation | 1,559 | Next 10% chronologically |
| Test | 1,560 | Last 10% chronologically |

The chronological split was pre-built in the NPZ by `extract_temporal_dataset.py`,
which sorted all sequences by instance-level timestamps before slicing.
No random shuffling was applied. Future observations do not contaminate training.

`shuffle=False` was explicitly set in `model.fit()` to preserve order within
the training split.

---

## 12. Feature Scaling

| Scaler | Applied To | Fit Data | File |
|--------|-----------|----------|------|
| StandardScaler | X (7 features, 2D reshaped) | X_train only | `feature_scaler.pkl` |
| MinMaxScaler | Y (target) | Y_train only | `target_scaler.pkl` |

Validation and test data were transformed using the **training-fitted** scalers.
No scaler was fitted on validation, test, or the complete dataset.

---

## 13. LSTM Architecture

```
Input  →  shape (5, 7)
           ↓
LSTM(64 units, return_sequences=True)
           ↓
Dropout(0.2)
           ↓
LSTM(32 units)
           ↓
Dropout(0.2)
           ↓
Dense(16, activation='relu')
           ↓
Dense(1, activation='linear')   ← MinMax-scaled target output
```

| Parameter | Value |
|-----------|-------|
| LSTM layers | 2 |
| LSTM units | 64 → 32 |
| Dropout | 0.20 per LSTM layer |
| Dense layers | 2 (16-ReLU → 1-Linear) |
| Output activation | Linear (regression) |
| Loss function | MSE |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Max epochs | 100 |
| Shuffle during training | False (chronological order preserved) |

---

## 14. Training Configuration

| Item | Value |
|------|-------|
| EarlyStopping monitor | val_loss |
| EarlyStopping patience | 10 |
| Restore best weights | True |
| ModelCheckpoint | val_loss (best only) |
| Random seed | 42 |
| Python seed | os.environ PYTHONHASHSEED=42 |
| NumPy seed | np.random.seed(42) |
| TensorFlow seed | tf.random.set_seed(42) |

Note: TensorFlow on GPU may not guarantee full determinism due to
non-deterministic CUDA operations. Results may vary slightly across runs.

---

## 15. Training History

| Metric | Value |
|--------|-------|
| Epochs trained | 30 |
| Best epoch | 20 |
| Best val_loss (MSE) | 0.000003 |
| Final train_loss | 0.000560 |
| Final val_loss | 0.000019 |

See `plots/training_validation_loss.png` for the full learning curve.

---

## 16. Final Metrics

All metrics are reported on the **original (unscaled) CPU fraction scale**.

### Train Set

| Metric | Value |
|--------|-------|
| MAE | 0.000594 |
| RMSE | 0.001951 |
| R² | 0.8432 |

### Validation Set

| Metric | Value |
|--------|-------|
| MAE | 0.000106 |
| RMSE | 0.000163 |
| R² | 0.9877 |

### Test Set (Final — evaluated once)

| Metric | Value |
|--------|-------|
| MAE | 0.000417 |
| RMSE | 0.000547 |
| R² | 0.9838 |

> **Note**: R² is the coefficient of determination — NOT accuracy.
> R² = 0.9838 means 98.4% of variance in the test target
> is explained by the model. This is the standard regression quality measure.

---

## 17. Error Analysis

### Residual Statistics (Test Set)

| Statistic | Value |
|-----------|-------|
| Mean residual | -0.000301 |
| Std of residuals | 0.000457 |
| MAE | 0.000417 |
| RMSE | 0.000547 |

`residual = actual − predicted`

See `plots/residual_distribution.png`.

---

## 18. Performance by delta-time Bin (Test Set)

The following bins use the `delta_t_sec` at the last step (step 4) of each
test sequence — the gap from the last known observation to the target.

| Temporal Gap Bin | N Samples | MAE | RMSE | R² | Notes |
|------------------|-----------|-----|------|-----|-------|
| ≤ 5 min         |    401 | 0.000420 | 0.000604 |  0.9882 | |
| 5–15 min        |   1024 | 0.000409 | 0.000524 |  0.9812 | |
| 15–30 min       |    135 | 0.000462 | 0.000534 |  0.9734 | |
| 30–60 min       |      0 | — | — | — | insufficient samples (<10) |


> Note: The 30–60 min bin is expected to have 0 samples because the NPZ
> enforces a maximum internal gap of ≤ 1800 seconds (30 minutes).

See `plots/error_by_delta_t.png`.

---

## 19. Residual Analysis

- Mean residual of -0.000301 indicates slight over-prediction
- Standard deviation of residuals: 0.000457
- The residual distribution plot (plots/residual_distribution.png) shows the
  spread and symmetry of prediction errors

---

## 20. Overfitting Analysis

| Split | MAE | RMSE | R² |
|-------|-----|------|----|
| Train | 0.000594 | 0.001951 | 0.8432 |
| Validation | 0.000106 | 0.000163 | 0.9877 |
| Test | 0.000417 | 0.000547 | 0.9838 |

Train–Validation R² gap: -0.1445  
Validation–Test R² gap: 0.0039

Assessment:
- If Train R² >> Val R²: overfitting is present
- If Val R² ≈ Test R²: generalization is consistent
- Dropout(0.2) on both LSTM layers was included to regularize
- EarlyStopping with patience=10 prevents excessive training

---

## 21. Limitations

1. **Borg workload traces are not live cloud-provider telemetry.** The dataset
   represents Google's internal cluster workload, not commercial cloud billing.

2. **Variable time intervals.** Without `delta_t_sec`, the LSTM treats all
   consecutive steps as equally spaced, which is incorrect for 97.83% of
   sequences. This is a known limitation accepted for this phase.

3. **delta_t_sec removed due to leakage.** The benefit of explicit temporal
   gap encoding is lost. A corrected extraction (shift deltas by one step)
   could recover this in future work without leakage.

4. **Sequence density varies.** Some workload instances contribute many
   sequences; others contribute none. The LSTM may over-represent high-activity
   instances.

5. **LSTM requires temporal continuity.** Performance may degrade for sequences
   with large, irregular gaps, which this architecture cannot fully model.

6. **CPU forecasting alone does not demonstrate cost savings.** Energy,
   memory, and network costs are not considered.

7. **No auto-scaling or production deployment claims are made.** Phase 3 is
   strictly model development and evaluation.

8. **TensorFlow GPU non-determinism.** If GPU acceleration is used, results
   may vary slightly across runs despite seeding.

---

## 22. Reference Comparison with RF/XGBoost (Phase 2B — Read Only)

> These results are included as reference only.  
> No retraining was performed. Phase 2B artifacts were not modified.  
> Final model selection will be conducted in Phase 4.

| Model | Test MAE | Test RMSE | Test R² |
|-------|----------|-----------|---------|
| Random Forest (Phase 2B) | 0.000615 | 0.001899 | 0.9071 |
| XGBoost (Phase 2B) | 0.000877 | 0.002290 | 0.8650 |
| **LSTM (Phase 3)** | **0.000417** | **0.000547** | **0.9838** |

> **Important note:** The tabular models (RF/XGBoost) and LSTM operate on
> different dataset representations:
> - RF/XGBoost: flat feature vectors (65,398 pairs, 18 features including delta_t_sec)
> - LSTM: 3D sequences (15,594 sequences, 7 features after leakage removal)
> Direct metric comparison must be interpreted with this difference in mind.

---

## 23. Phase 3 Conclusion

Phase 3 has successfully:

- ✅ Loaded and inspected `borg_lstm_sequences.npz`
- ✅ Verified dataset dimensions and feature ordering
- ✅ Audited `delta_t_sec` for temporal leakage (confirmed → removed)
- ✅ Validated all 15,594 sequences (NaN/Inf free, gap constraints satisfied)
- ✅ Verified chronological split (no data leakage across splits)
- ✅ Applied scalers fitted exclusively on training data
- ✅ Built and trained a 2-layer stacked LSTM (64 → 32 units)
- ✅ Applied EarlyStopping and ModelCheckpoint
- ✅ Evaluated on train/validation/test with MAE, RMSE, R²
- ✅ Performed error analysis by temporal gap bin
- ✅ Generated all required plots
- ✅ Saved all artifacts

**PHASE 3 STATUS: COMPLETE**

---

## 24. Recommended Next Step

Proceed to **Phase 4: Final Model Comparison and Selection**.

Phase 4 should:
- Load Phase 2B RF and XGBoost models
- Load Phase 3 LSTM model
- Apply fair evaluation on the same test data (where possible)
- Compare MAE, RMSE, R² across all three models
- Discuss trade-offs (inference speed, interpretability, dataset size)
- Select the best model with documented justification
- Document the final model selection rationale for the PBL viva

---

*Phase 3 Report — CloudScale AI PBL Project*  
*Generated: 2026-10-05 11:07:45*
