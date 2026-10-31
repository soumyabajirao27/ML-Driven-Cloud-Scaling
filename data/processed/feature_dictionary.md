# CloudScale AI — Borg Workload Feature & Leakage Dictionary

This document details all columns present in `borg_cloudscale_processed.csv`, categorized by availability and predictive leakage risk.

## 1. Predictive Input Features (Available at or before Prediction Time $t$)

| Feature Name | Type | Category | Semantic Description | Predictive Role |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp_us` | float | Temporal | Event timestamp in microseconds ($10^{-6}$ sec) | Time Index |
| `time_hours` | float | Temporal | Timestamp converted to hours relative to trace start | Time Feature |
| `time_days` | float | Temporal | Timestamp converted to days relative to trace start | Time Feature |
| `priority` | float | Categorical | Workload priority level (0 to 450) | **Valid Input** |
| `scheduling_class` | float | Categorical | Scheduling tier (0=non-prod, 1=batch, 2=mid, 3=prod) | **Valid Input** |
| `collection_type` | float | Categorical | Collection type code | **Valid Input** |
| `instance_events_type` | float | Categorical | Instance event type code | **Valid Input** |
| `collections_events_type` | float | Categorical | Collection event type code | **Valid Input** |
| `cluster` | int | Categorical | Cluster identifier | **Valid Input** |
| `vertical_scaling` | float | Policy Flag | Vertical auto-scaling policy indicator | **Valid Input** |
| `scheduler` | float | Policy Flag | Borg scheduler policy type | **Valid Input** |
| `cpu_request` | float | Numerical | Requested CPU capacity (normalized node fraction) | **Valid Input** |
| `memory_request` | float | Numerical | Requested RAM capacity (normalized node fraction) | **Valid Input** |
| `assigned_memory` | float | Numerical | Memory assigned by Borg master | **Valid Input** |
| `cpi` | float | Numerical | Cycles Per Instruction hardware telemetry | **Valid Input** |
| `mai` | float | Numerical | Memory Accesses Per Instruction hardware telemetry | **Valid Input** |
| `sample_rate` | float | Numerical | Metric collection sample rate | **Valid Input** |

---

## 2. Metadata Identifiers (Excluded from ML Model Inputs to Prevent Memorization)

| Feature Name | Type | Category | Reason for Exclusion |
| :--- | :--- | :--- | :--- |
| `collection_id` | string | Identifier | Overlaps between Train and Evaluation splits; causes job memorization. |
| `instance_index` | int | Identifier | Specific instance index within collection. |
| `machine_id` | string | Identifier | Overlaps between Train and Evaluation splits; causes node memorization. |
| `alloc_collection_id` | string | Identifier | Parent allocation identifier. |

---

## 3. Post-Event Analysis Telemetry (EXCLUDED from Predictive Model Inputs — Contains Future Information / Target Leakage)

| Feature Name | Type | Reason for Exclusion from Inputs |
| :--- | :--- | :--- |
| `duration_sec` | float | Total execution duration `(end_time - start_time) / 1e6` — Observed ONLY AFTER run completes. |
| `event` | string | Termination event code (`FINISH`, `FAIL`, `EVICT`) — Observed ONLY AT EXIT. |
| `failed` | int | Binary failure flag — Observed ONLY AT EXIT. |
| `cpu_avg_usage` | float | Interval average CPU usage — Direct Target Component / Post-event telemetry. |
| `memory_avg_usage` | float | Interval average RAM usage — Post-event telemetry. |
| `cpu_max_usage` | float | Interval peak CPU usage — Observed ONLY AFTER interval finishes. |
| `memory_max_usage` | float | Interval peak RAM usage — Observed ONLY AFTER interval finishes. |
| `cpu_rand_sample` | float | Random usage sample — Post-event telemetry. |
| `page_cache_memory` | float | Linux page cache memory — Post-event telemetry. |
| `cpu_dist_*` | float | Usage distribution statistics — Observed ONLY AFTER interval finishes. |
| `tail_dist_*` | float | Tail usage distribution statistics — Observed ONLY AFTER interval finishes. |
| `cpu_utilization_pct` | float | Derived instance CPU utilization — Direct Target Component (`(cpu_avg / cpu_req) * 100`). |
| `memory_utilization_pct` | float | Derived instance RAM utilization — Post-event telemetry. |
| `cpu_burst_factor` | float | Burstiness ratio — Post-event telemetry. |
| `memory_headroom` | float | Unused assigned memory margin — Post-event telemetry. |
| `page_cache_ratio` | float | Page cache ratio — Post-event telemetry. |
| `cpu_oversubscription` | float | Excess CPU demand — Post-event telemetry. |
| `utilization_state` | string | Baseline state (`OVERLOADED`, `UNDERUTILIZED`, `OPTIMAL`) — **Target Classification Label ONLY**. |

---

## 4. Target Declarations (Audit Flags)

| Target Field | Type | Description | Scientific Validity |
| :--- | :--- | :--- | :--- |
| `same_row_current_cpu_avg` | float | Same-row CPU usage ($t=0$) | **Current-State Telemetry (Leakage if input contains same-row usage)** |
| `same_row_current_cpu_util` | float | Same-row CPU utilization % ($t=0$) | **Current-State Telemetry (Leakage if input contains same-row usage)** |
| `has_future_target` | int | Flag indicating valid $t 	o t+1$ pair ($\Delta t \le 1 	ext{ hr}$) | `0` across 30k slice due to time gap sampling |
| `target_future_cpu_avg` | float | Future CPU usage ($t + 1$) | `NaN` on 30k slice; requires full trace extraction |
| `target_future_cpu_util` | float | Future CPU utilization % ($t + 1$) | `NaN` on 30k slice; requires full trace extraction |
