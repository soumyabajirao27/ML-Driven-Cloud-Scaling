# CloudScale AI: Machine Learning-Driven Auto Scaling and Cost Optimization Framework for Cloud Resources

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6.svg)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg)](https://www.postgresql.org/)
[![Build & Tests](https://img.shields.io/badge/Tests-81%2F81%20Passed-brightgreen.svg)]()

---

## 1. Project Overview

**CloudScale AI** is an intelligent, machine-learning-driven autoscaling and FinOps cost optimization framework designed to mitigate cloud over-provisioning and SLA latency spikes. By coupling real-time telemetry from Google Borg historical traces with supervised machine learning and a stateful proactive control policy (**Policy B**), the system generates explainable right-sizing and capacity-scaling recommendations with hysteresis deadbands and cooldown dampening.

---

## 2. Problem Statement & Motivation

Modern hyperscale cloud architectures frequently suffer from two opposing operational pathologies:
1. **Severe Over-Provisioning**: Infrastructure engineers commonly allocate 3× to 5× peak capacity to avoid downtime, leading to excessive idle power draw, carbon footprint, and wasted cloud spend.
2. **Reactive Auto-Scaling Lag**: Native threshold-based reactive scaling rules (e.g., AWS CloudWatch alarms, Kubernetes HPA) only trigger *after* high utilization thresholds are crossed. Because VM and container provisioning takes 60–300 seconds, workloads frequently encounter severe latency spikes and SLO breaches during the provisioning lag.

**CloudScale AI** addresses this challenge through **proactive demand forecasting**: predicting future resource utilization one interval ahead to scale capacity *prior* to overload conditions, while enforcing strict cooldown and hysteresis dampening to prevent system thrashing.

---

## 3. Key Objectives

- **Telemetry Integration**: Ingest real-world Google Borg trace telemetry into a relational database with strict source isolation and chronological ordering.
- **Supervised ML Forecasting**: Train and evaluate machine learning models (Random Forest, XGBoost, LSTM) on temporal workload sequences without target leakage.
- **Policy B Decision Engine**: Implement an optimization policy that combines proactive forecast thresholds with hysteresis buffers and cooldown safeguards.
- **Simulated FinOps & Power Modeling**: Model normalized cloud costs and linear server electrical power/energy consumption (100W Idle – 250W Max).
- **Interactive Enterprise UI**: Expose explainable decisions, demand forecasts, and FinOps telemetry through a modern React dashboard.

---

## 4. System Architecture

CloudScale AI is structured into four decoupled layers:

```
[ Frontend: React 18 + Vite + Tailwind ]
                   ↕ (REST JSON / JWT Bearer)
[ Backend: FastAPI + Pydantic v2 + SQLAlchemy 2.0 ]
       ↕                                  ↕
[ Persistence: PostgreSQL 16 ]     [ ML Models: Scikit-Learn RandomForest ]
                                          ↕
                             [ Telemetry: Google Borg Traces ]
```

Detailed diagrams and sequence flows are documented in [`docs/architecture.md`](docs/architecture.md).

---

## 5. Technology Stack

- **Backend**: FastAPI 0.115, Python 3.10, Pydantic v2, SQLAlchemy 2.0, Psycopg 3.
- **Machine Learning**: Scikit-Learn 1.5, Joblib, NumPy.
- **Database**: PostgreSQL 16 with Alembic schema migrations.
- **Frontend**: React 18.3, Vite 5.4, TypeScript 5.5, Tailwind CSS, Recharts, Lucide Icons.
- **Dataset**: Google Cluster Workload Traces 2019 (Borg temporal trace pairs).

---

## 6. Dataset & Feature Engineering

### Dataset Specifications
- **Origin**: Google Borg Cluster Workload Traces (2019).
- **Primary Source**: `data/temporal/borg_temporal_pairs.csv` (65,398 samples) and `data/processed/borg_cloudscale_processed.csv`.
- **Chronological Window**: Consecutive 5-minute telemetry intervals.

### Feature Pipeline (9 Features)
Predictions are generated strictly from causal past observations to eliminate future target leakage:
1. `cpu_utilization` ($t$)
2. `memory_utilization` ($t$)
3. `storage_utilization` ($t$)
4. `network_utilization` ($t$)
5. `cpu_rolling_mean_3` (3-sample moving average)
6. `memory_rolling_mean_3` (3-sample moving average)
7. `cpu_lag_1` ($t-1$ CPU sample)
8. `memory_lag_1` ($t-1$ Memory sample)
9. `hour_of_day` (0–23 UTC timestamp extract)

---

## 7. Machine Learning Model & Experimental Results

### Offline Evaluation Summary (Chronological 90/10 Split)
Models were evaluated on 65,398 chronological temporal samples from Google Borg traces:

| Model Architecture | Input Dimension | Target | Test MAE | Test RMSE | Test $R^2$ | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Naive Baseline** | 1 (`cpu_avg_usage_t`) | Future CPU Avg | 0.000936 | 0.003713 | 0.6449 | Baseline |
| **Phase 2B Random Forest** | 18 features | Future CPU Avg | 0.000615 | 0.001899 | 0.9071 | Offline Benchmark |
| **Phase 3 LSTM** | 7 features | Future CPU Avg | 0.000417 | 0.000547 | 0.9838 | Research Benchmark |
| **Production Random Forest** | 9 features | Next-sample CPU % | 4.50% | 6.20% | 0.9420 | **Live Inference Engine** |

*Note*: The 9-feature `RandomForestRegressor` (`demand_predictor.joblib`) was selected for live application inference due to its runtime stability, zero deep-learning external dependencies, and direct compatibility with relational database metrics.

---

## 8. Optimization Policy (Policy B)

**Policy B** evaluates forecasted future demand to recommend proactive scaling:
- **Proactive Scale-Up**: Triggered when predicted $\text{CPU} > 87.0\%$ ($85\% + 2\%$ hysteresis). Capacity is doubled (clamped to max 32 Units).
- **Proactive Scale-Down**: Triggered when predicted $\text{CPU} < 18.0\%$ ($20\% - 2\%$ hysteresis). Capacity is reduced by $50\%$ (clamped to min 1 Unit).
- **Neutral Band (Maintain)**: Maintained when $18.0\% \le \text{Predicted CPU} \le 87.0\%$.
- **Cooldown Safeguard**: Enforces a 2-step / 300-second lock following scaling actions to damp rapid oscillations. Subsequent requests during cooldown return `DEFERRED_COOLDOWN`.

---

## 9. Cost and Power Estimation Formulas

1. **Normalized Cloud Cost**:
   $$\text{Cost} = \text{Capacity}_{\text{cpu}} \times 1.0 + \text{Utilization}_{\text{mem}} \times 0.5$$
2. **Server Electrical Power Model**:
   $$P(t) = P_{\text{idle}} + \text{Utilization} \times (P_{\text{max}} - P_{\text{idle}})$$
   - $P_{\text{idle}} = 100\text{ W}$, $P_{\text{max}} = 250\text{ W}$.
3. **Energy Consumption**:
   $$\text{Energy (Wh)} = P(t) \times \left(\frac{300\text{ s}}{3600\text{ s}}\right)$$

---

## 10. Installation & Setup Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- PostgreSQL 14+ running locally on port 5432

### 1. Database Configuration
Create the PostgreSQL database and user:
```sql
CREATE DATABASE cloudscale;
CREATE USER postgres WITH PASSWORD 'root123';
GRANT ALL PRIVILEGES ON DATABASE cloudscale TO postgres;
```

### 2. Backend Setup
```powershell
# Navigate to backend
Set-Location d:\PBL\backend

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run database migrations
alembic upgrade head
```

### 3. Frontend Setup
```powershell
# Navigate to frontend
Set-Location d:\PBL\frontend

# Install dependencies (if not installed)
npm install
```

---

## 11. Running the Application

### Start Backend API Server
```powershell
Set-Location d:\PBL\backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at: `http://localhost:8000/docs`

### Start Frontend Client
```powershell
Set-Location d:\PBL\frontend
npm run dev
```
Dashboard will be live at: `http://localhost:5173`

---

## 12. Executing Automated Test Suites

All 81 automated tests across Stages 1–4 can be executed using the backend virtual environment:

```powershell
Set-Location d:\PBL\backend

# Run Stage 1: Google Borg Integration & Ingestion Tests (8 tests)
.\venv\Scripts\python.exe tests\test_stage1_borg.py

# Run Stage 2: ML Model Validation & Provenance Tests (26 tests)
.\venv\Scripts\python.exe tests\test_stage2_ml.py

# Run Stage 3: Policy B Autoscaling, Cooldown & Cost/Power Tests (28 tests)
.\venv\Scripts\python.exe tests\test_stage3_policy_b.py

# Run Stage 4: End-to-End Pipeline & Security Verification (19 tests)
.\venv\Scripts\python.exe tests\test_stage4_e2e_verification.py
```

### Verify Frontend Production Build
```powershell
Set-Location d:\PBL\frontend
npm run build
```

---

## 13. Step-by-Step Evaluator Demonstration Script

Follow this reproducible walkthrough to demonstrate the complete framework:

1. **Start Services**: Launch FastAPI backend on port 8000 and Vite frontend on port 5173.
2. **Access Dashboard**: Open browser to `http://localhost:5173` and log in (`admin@test.com` / `admin123` or demo service account `svc_test@test.com` / `svc123`).
3. **Inspect Connected Resources**:
   - Navigate to **Resources**.
   - Observe both `mock` provider instances and `google_borg` ingested instances (e.g. `borg-116646-17`).
4. **Trigger Bounded Google Borg Replay**:
   - Click **Replay Borg** button (or execute `POST /api/v1/resources/replay-borg?max_intervals=100`).
   - Confirm idempotency: duplicate intervals are skipped; imported record statistics are reported.
5. **View ML Demand Forecasts**:
   - Navigate to **AI Predictions**.
   - Select resource `borg-116646-17`.
   - Inspect the forecasted CPU utilization curve and model provenance badge (`RandomForestRegressor`).
6. **Evaluate Policy B Scaling Recommendations**:
   - Navigate to **Recommendations**.
   - Inspect the recommendation card for `borg-116646-17`.
   - Notice the **Policy B (Proactive ML)** tag, safeguard status (`Normal`), target capacity, and decision reasoning.
7. **Demonstrate Cooldown Safeguard**:
   - Trigger repeated rapid optimization analyses via API (`POST /api/v1/resources/{id}/analyze`).
   - Observe that the action immediately transitions to `DEFERRED_COOLDOWN` with remaining step counts clearly displayed.
8. **Inspect FinOps & Energy Modeling**:
   - Navigate to **Cost Optimization**.
   - Inspect normalized monthly cost figures and linear power draw calculations.
9. **Run Automated Test Verification**:
   - Execute all 4 test suites in terminal to demonstrate 81/81 passing unit, regression, and integration tests.

---

## 14. Known Limitations & Architectural Scope

- **Simulated Provider Execution**: Scaling actions are simulated recommendations; no real cloud hypervisor or cloud provider APIs (AWS EC2 / GCP Compute Engine) are mutated.
- **In-Memory Cooldown State**: The `ScalingStateManager` uses process-level `threading.RLock`. Multi-worker cluster deployments would require a Redis-backed distributed lock.
- **Historical Telemetry Traces**: Google Borg metrics represent historical trace observations rather than live streaming telemetry.
- **Estimated Cost & Power**: Power draws are derived from an established 100W–250W server linear mathematical model rather than physical hardware wattmeter readings.

---

## 15. Future Scope

- Integration with real Kubernetes custom metrics APIs via KEDA.
- Distributed cooldown coordination using Redis pub/sub.
- Quantile regression for true statistical confidence intervals instead of heuristic bands.
- Multi-cloud billing API sync (AWS Cost Explorer, GCP Cloud Billing APIs).
