# CloudScale AI — Full Borg Trace Temporal Feasibility Report (Phase 2A - Final Validation)

**Generated Date**: 2026-10-04 19:10:29  
**Primary Dataset**: `d:\PBL\borg_traces_data.csv` (313.10 MB, 405,894 rows)  
**Subset Comparison**: `d:\PBL\borg_traces_30000.csv` (30,000 rows)  
**Extracted Temporal Dataset**: `d:\PBL\data\temporal\borg_temporal_pairs.csv`  
**Extracted LSTM NPZ Archive**: `d:\PBL\data\temporal\borg_lstm_sequences.npz`  

---

## 1. Executive Summary & Temporal Regularity Audit

This final validation audited time-step regularity, sequence validity, time-gap feature inclusion, and target horizons across the extracted temporal datasets (`borg_temporal_pairs.csv` and `borg_lstm_sequences.npz`).

### Audit Findings:
1. **Time-Step Regularity Audit**:
   - Out of 15,594 strict sliding sequences ($L=5$, step gap $\le 30\text{ min}$), **2.17%** have constant uniform time gaps ($\Delta t \le 60\text{ s}$), while **97.83%** have irregular interval spacing (ranging from $3.2\text{ min}$ to $14.6\text{ min}$).
   - **Scientific Directive**: Standard Vanilla LSTM assuming uniform spacing is inappropriate. **VALID ONLY WITH TIME-GAP FEATURES (Option B)**.
   - **Resolution Applied**: Incorporated `delta_t_sec` as explicit Feature #8 in the 3D sequence array tensor $X \in [N, 5, 8]$.

2. **Target Horizon Classification**:
   - Both RF/XGBoost pairs and LSTM sequences represent a **Variable-Gap 1-Step Future Horizon ($t \to t+\Delta t$)** where median horizon $\Delta t = 6.3\text{ min}$ for LSTM sequences and $10.5\text{ min}$ for Tabular pairs.
   - It is explicitly classified as a **Variable-Horizon Forecasting Task**.

3. **Train / Validation / Test Splitting**:
   - Chronological split applied cleanly across 80% Train, 10% Validation, and 10% Test sets.

---

## 2. Quantitative Summary Statistics

| Metric Name | Empirical Finding |
| :--- | :--- |
| **Full Dataset Raw Rows** | `405,894` |
| **Full Dataset Valid Timestamp Rows** | `405,891` |
| **Full Dataset Columns** | `34` |
| **Temporal Coverage** | `31.01` days |
| **Unique Collections (`collection_id`)** | `4,057` |
| **Unique Instances (`collection_id, instance_index`)** | `242,943` |
| **Unique Machines (`machine_id`)** | `96,174` |
| **Extracted 1-Step Future Target Pairs ($0 < \Delta t \le 1\text{ hr}$)** | `65,398` |
| **1-Step Pairs Train Set (80%)** | `52,318` |
| **1-Step Pairs Validation Set (10%)** | `6,540` |
| **1-Step Pairs Test Set (10%)** | `6,540` |
| **Extracted Strict LSTM Sequences ($L=5$, gap $\le 30\text{ min}$)** | `15,594` |
| **LSTM Input Tensor Shape ($X$)** | `(15,594, 5, 8)` |
| **LSTM Time-Gap Feature** | Included (`delta_t_sec` at index #7) |
| **Best-Supported Forecasting Horizon** | Variable 1-Step Horizon ($0 < \Delta t \le 30\text{ min}$, median gap: $6.3\text{ min}$) |
| **Recommended Workload Grouping Key** | `(collection_id, instance_index)` |

---

## 3. Final Model-Training Gate Evaluation

- **RF/XGBOOST DATASET (`borg_temporal_pairs.csv`)**: **READY**
- **LSTM DATASET (`borg_lstm_sequences.npz`)**: **READY**
- **Sequence Recommendation**: **VALID ONLY WITH TIME-GAP FEATURES (Option B)**

---

## 4. Confirmation Checklist

- [x] `d:\PBL\borg_traces_data.csv` was NOT modified (Read-Only).
- [x] `d:\PBL\borg_traces_30000.csv` was NOT modified.
- [x] Existing backend was NOT modified.
- [x] Existing frontend was NOT modified.
- [x] Existing database schema was NOT modified.
- [x] Existing synthetic/mock ML pipeline was NOT modified.
- [x] Existing APIs were NOT modified.
- [x] No model was trained in this phase.
- [x] No fake performance metrics were generated.
- [x] No fake temporal observations were created.

---

### Final Decision Recommendation

Proceed to Phase 2B
