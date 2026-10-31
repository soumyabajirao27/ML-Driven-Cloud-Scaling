# CloudScale AI — Borg Workload Trace Data Quality & Target Leakage Audit Report

**Generated Date**: 2026-10-04 18:59:20  
**Primary Working Subset**: `d:\PBL\borg_traces_30000.csv`  
**Processed Output Dataset**: `d:\PBL\data\processed\borg_cloudscale_processed.csv`  
**Audit Status**: **PHASE 1 NOT YET APPROVED — CORRECTIONS REQUIRED**  

---

## 1. Executive Summary & Audit Purpose

This report provides a strict scientific audit of target definition, temporal prediction validity, feature availability, and data leakage across the 30,000-row subset of the Google Borg workload trace dataset (`borg_traces_30000.csv`).

Key Audit Finding:
The initial target definition derived `target_cpu_avg_usage` and `target_cpu_util_pct` from the **exact same row ($t=0$)** as the telemetry input features. Using same-row telemetry to predict same-row targets constitutes **100% direct target leakage** (current-state reconstruction) rather than future prediction. Furthermore, 93.02% of workload instance keys appear only once in this 30,000-row slice, meaning **no valid consecutive $t \to t+1$ time-series prediction pairs (\Delta t \le 1 \text{ hour})** exist in this subset.

---

## 2. Quantitative Summary Statistics

| Metric | Empirical Value | Audit Assessment |
| :--- | :--- | :--- |
| **Original Row Count** | `30,000` | Validated |
| **Processed Row Count** | `30,000` | Validated |
| **Rows Removed** | `0` | Validated |
| **Original Column Count** | `34` | Validated |
| **Processed Feature Count** | `48` | Categorized into pre-prediction vs post-event |
| **Duplicate Exact Rows** | `0` | Validated |
| **Duplicate IDs** | `0` | Validated |
| **Same-Row Target Leakage** | **100% Leakage if same-row telemetry is passed as input** | **CRITICAL AUDIT LIMITATION** |
| **Valid Consecutive $t \to t+1$ Pairs (\Delta t \le 1 \text{ hr})** | `0` | **CRITICAL AUDIT LIMITATION** |
| **Train Set Rows (80%)** | `24,000` | Chronological split applied |
| **Validation Set Rows (10%)** | `3,000` | Chronological split applied |
| **Test Set Rows (10%)** | `3,000` | Chronological split applied |

---

## 3. Target Generation Logic & Transformation Audit

In the initial script, targets were created via:
```python
target_cpu_avg_usage = rec['cpu_avg_usage']  # Derived from row['average_usage']['cpus']
target_cpu_util_pct = (cpu_avg / cpu_req) * 100  # Derived from same row
```

### Scientific Target Evaluation:
1. **Derived From**: `average_usage` and `resource_request` of the **current observation ($t$)**.
2. **Temporal Alignment**: Current row ($t=0$).
3. **Prediction Horizon \Delta t**: $0 \text{ seconds}$ (Same-row observation).
4. **Input Feature Overlap**: Features like `cpu_avg_usage`, `cpu_max_usage`, `cpu_utilization_pct`, `cpu_burst_factor`, and `cpu_dist_median` contain the target value or its components directly.
5. **Leakage Determination**: **DIRECT TARGET LEAKAGE**. Any regression model given `cpu_avg_usage` as an input to predict `same_row_current_cpu_avg` achieves trivial ~1.0 $R^2$ by current-state copying, not forecasting.

---

## 4. Target & Feature Leakage Audit Matrix

| Feature Name | Relationship to Target | Available Before Prediction ($t$)? | Leakage Risk if Predicting Usage? | Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| `priority` | Static Specification | **Yes** | None | **KEEP** (Predictive Feature) |
| `scheduling_class` | Static Specification | **Yes** | None | **KEEP** (Predictive Feature) |
| `collection_type` | Static Specification | **Yes** | None | **KEEP** (Predictive Feature) |
| `instance_events_type` | Event Metadata | **Yes** | None | **KEEP** (Predictive Feature) |
| `collections_events_type` | Event Metadata | **Yes** | None | **KEEP** (Predictive Feature) |
| `cluster` | Topology Metadata | **Yes** | None | **KEEP** (Predictive Feature) |
| `vertical_scaling` | Policy Metadata | **Yes** | None | **KEEP** (Predictive Feature) |
| `scheduler` | Policy Metadata | **Yes** | None | **KEEP** (Predictive Feature) |
| `cpu_request` | Allocation Specification | **Yes** | None | **KEEP** (Predictive Feature) |
| `memory_request` | Allocation Specification | **Yes** | None | **KEEP** (Predictive Feature) |
| `assigned_memory` | Master Allocation | **Yes** | None | **KEEP** (Predictive Feature) |
| `cpi` | Hardware Metric | **Yes** (if sampled) | Low | **KEEP** (Predictive Feature) |
| `mai` | Hardware Metric | **Yes** (if sampled) | Low | **KEEP** (Predictive Feature) |
| `end_time` | Interval Termination | **No** (Observed after run) | **HIGH** | **EXCLUDE from inputs** |
| `duration_sec` | Total Execution Time | **No** (Observed after run) | **HIGH** | **EXCLUDE from inputs** |
| `event` | Termination Status | **No** (Observed at exit) | **HIGH** | **EXCLUDE from inputs** |
| `failed` | Failure Indicator | **No** (Observed at exit) | **HIGH** | **EXCLUDE from inputs** |
| `cpu_avg_usage` | Usage Telemetry | **No** (Observed during/after) | **DIRECT LEAKAGE** | **EXCLUDE from inputs** |
| `memory_avg_usage` | Usage Telemetry | **No** (Observed during/after) | **HIGH** | **EXCLUDE from inputs** |
| `cpu_max_usage` | Peak Telemetry | **No** (Observed after interval) | **HIGH** | **EXCLUDE from inputs** |
| `memory_max_usage` | Peak Telemetry | **No** (Observed after interval) | **HIGH** | **EXCLUDE from inputs** |
| `cpu_utilization_pct` | Derived Telemetry | **No** (Contains `cpu_avg`) | **DIRECT LEAKAGE** | **EXCLUDE from inputs** |
| `memory_utilization_pct` | Derived Telemetry | **No** (Contains `mem_avg`) | **HIGH** | **EXCLUDE from inputs** |
| `cpu_burst_factor` | Derived Telemetry | **No** (Contains `cpu_max`) | **HIGH** | **EXCLUDE from inputs** |
| `memory_headroom` | Derived Telemetry | **No** (Contains `mem_avg`) | **HIGH** | **EXCLUDE from inputs** |
| `page_cache_ratio` | Derived Telemetry | **No** (Contains `mem_avg`) | **HIGH** | **EXCLUDE from inputs** |
| `cpu_oversubscription` | Derived Telemetry | **No** (Contains `cpu_max`) | **HIGH** | **EXCLUDE from inputs** |
| `utilization_state` | Derived Label | **No** (Contains `cpu_util`) | **DIRECT LEAKAGE** | **Target Class Only** |

---

## 5. Distinction of Three Prediction Tasks

To prevent scientific ambiguity, CloudScale AI distinguishes three tasks:

### Task A: Current-State Telemetry Reconstruction
- **Objective**: Reconstruct `cpu_avg_usage` from hardware telemetry (`cpi`, `mai`, `assigned_memory`).
- **Status on 30k Subset**: Feasible, but has low operational utility for auto-scaling.

### Task B: Future CPU Time-Series Forecasting ($CPU(t + \Delta t)$)
- **Objective**: Predict $CPU(t + 1)$ given past telemetry $CPU(t)$ using only features available at or before $t$.
- **Status on 30k Subset**: **NOT FEASIBLE**. Out of 24,467 instances, 93.02% appear only ONCE in this random 30k slice. Zero contiguous $t \to t+1$ pairs with \Delta t \le 1 \text{ hour} exist in this subset.
- **Requirement**: Extract contiguous trace sequences from the full 328 MB dataset (`borg_traces_data.csv`).

### Task C: Workload Capacity & Policy Classification
- **Objective**: Predict workload `utilization_state` (`UNDERUTILIZED`, `OPTIMAL`, `OVERLOADED`) using strictly pre-execution specifications (`cpu_request`, `memory_request`, `priority`, `scheduling_class`, `collection_type`, `cluster`, `assigned_memory`).
- **Status on 30k Subset**: **FEASIBLE for Tabular ML (Random Forest, XGBoost)** after completely removing post-event telemetry (`cpu_avg_usage`, `cpu_max_usage`, `duration_sec`, `event`, `failed`) from input features.

---

## 6. Global Chronological Split & Overlap Audit

- **Train Set (80%)**: `24,000` rows | Timestamp Range: `0` us to `1970569814690` us
- **Validation Set (10%)**: `3,000` rows | Timestamp Range: `1970589936707` us to `2342365354408` us
- **Test Set (10%)**: `3,000` rows | Timestamp Range: `2342483652289` us to `2678923967375` us

### Identifier Overlap Audit:
| Identifier Level | Unique Train | Unique Val | Unique Test | Val Overlap with Train | Test Overlap with Train | Potential Leakage Risk | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `collection_id` | 1,657 | 302 | 334 | 174 (57.6%) | 169 (50.6%) | Memorization of specific job IDs | **EXCLUDE `collection_id` from model inputs** |
| `machine_id` | 21,640 | 2,942 | 2,942 | 497 (16.9%) | 511 (17.4%) | Memorization of node IDs | **EXCLUDE `machine_id` from model inputs** |
| `instance_index` | 19,489 | 2,693 | 2,764 | 239 (8.9%) | 216 (7.8%) | Memorization of instance indices | **EXCLUDE `instance_index` from model inputs** |

---

## 7. Model Suitability Evaluation

### A. LSTM Suitability:
- **Evaluation**: **INSUFFICIENT**. The 30,000-row sampled subset does not contain continuous time sequences per instance. 93.02% of instance keys appear only once.
- **Statement**: *"The current 30,000-row subset is not sufficient for robust sequence forecasting at the chosen workload grouping level."*
- **Recommendation**: Extract continuous time-series traces from `borg_traces_data.csv` for the future LSTM phase.

### B. Tabular ML Suitability (Random Forest & XGBoost):
- **Dataset Structure**: Suitable.
- **Feature Suitability**: Suitable ONLY after excluding post-event usage telemetry (`cpu_avg_usage`, `cpu_max_usage`, `duration_sec`, `event`, `failed`, `cpu_dist_*`).
- **Target Suitability**: Suitable ONLY for Task C (Pre-execution Workload Utilization State Classification) or when trained on contiguous $t \to t+1$ target pairs extracted from full trace data.
- **Prediction Task**: Task C (Pre-execution Classification) is structurally valid on the 30k subset.

---

## 8. Confirmation Checklist

- [x] `d:\PBL\borg_traces_30000.csv` was NOT modified.
- [x] `d:\PBL\borg_traces_data.csv` was NOT modified.
- [x] Existing backend was NOT modified.
- [x] Existing frontend was NOT modified.
- [x] Existing database schema was NOT modified.
- [x] Existing synthetic/mock ML pipeline was NOT modified.
- [x] Existing production ML behavior was NOT modified.
- [x] No model was trained in this phase.
- [x] No fake performance metrics were generated.

---

### Audit Conclusion & Status Directive

PHASE 1 NOT YET APPROVED — CORRECTIONS REQUIRED
