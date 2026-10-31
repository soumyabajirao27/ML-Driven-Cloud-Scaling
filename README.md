# CloudScale AI: Machine Learning-Driven Auto Scaling and Cost Optimization Framework for Cloud Resources

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B-4169E1.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

An evaluation-ready, full-stack research prototype for cloud resource optimization. CloudScale AI couples machine learning-driven workload demand prediction trained on empirical **Google Borg cluster traces** with a closed-loop reactive/predictive autoscaling engine (**Policy B**). The system computes granular, simulated cost and server energy consumption estimates under varying workload conditions, surfacing actionable scaling decisions through an interactive React dashboard.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Proposed Solution](#3-proposed-solution)
- [4. Key Features](#4-key-features)
- [5. System Architecture](#5-system-architecture)
- [6. Technology Stack](#6-technology-stack)
- [7. Machine Learning Methodology](#7-machine-learning-methodology)
- [8. Dataset & Telemetry Ingestion](#8-dataset--telemetry-ingestion)
- [9. Autoscaling Strategy (Policy B)](#9-autoscaling-strategy-policy-b)
- [10. Cost & Energy Optimization Models](#10-cost--energy-optimization-models)
- [11. Model Evaluation & Benchmark Results](#11-model-evaluation--benchmark-results)
- [12. Project Directory Structure](#12-project-directory-structure)
- [13. Prerequisites](#13-prerequisites)
- [14. Installation & Local Setup](#14-installation--local-setup)
- [15. Configuration & Environment Variables](#15-configuration--environment-variables)
- [16. REST API Documentation](#16-rest-api-documentation)
- [17. Verification, Testing & Code Quality](#17-verification-testing--code-quality)
- [18. User Interface & Dashboards](#18-user-interface--dashboards)
- [19. System Limitations & Assumptions](#19-system-limitations--assumptions)
- [20. Contributing Guidelines](#20-contributing-guidelines)
- [21. License](#21-license)
- [22. Authors & Acknowledgments](#22-authors--acknowledgments)

---

## 1. Project Overview

Cloud infrastructure management faces an inherent tension between **over-provisioning** (allocating static capacity for peak loads, yielding idle resource waste and inflated operational expenditure) and **under-provisioning** (allocating lean capacity, triggering Service Level Agreement violations and application throttling during demand spikes).

**CloudScale AI** resolves this trade-off by combining:
1. **Predictive Workload Modeling**: A trained `RandomForestRegressor` (`model_id: random_forest_v1`) operating on rolling historical utilization metrics derived from Google Borg cluster traces.
2. **Policy B Autoscaling**: A dual-threshold heuristic with dynamic hysteresis bands, safety bounds, and step/time-based cooldown guards to prevent scaling oscillations (flapping).
3. **Simulated Cost & Power Modeling**: Rigorous analytical cost estimation (hourly VM instance pricing) and server power modeling based on established linear chassis power functions ($P(u) = 100\text{ W} + 150\text{ W} \times u_{\text{norm}}$).
4. **Transparent Evaluation & Data Governance**: Complete separation between historical Google Borg cluster traces and simulated runtime cloud resources, with end-to-end verification across 81 automated assertions.

---

## 2. Problem Statement

Modern enterprise and cloud applications experience highly dynamic, non-linear traffic patterns. Conventional autoscaling mechanisms (e.g., Kubernetes Horizontal Pod Autoscaler default rules or AWS CloudWatch metric alarms) rely primarily on reactive static thresholds:
- **Reactive Lag**: Scaling triggers only *after* resource utilization breaches an upper threshold, causing SLA degradations while new instances initialize and join the load balancer.
- **Resource Flapping**: Transient utilization spikes often trigger rapid succession of scale-up and scale-down events, generating cluster instability and hypervisor overhead.
- **Cost and Carbon Blindness**: Traditional infrastructure scaling decisions operate purely on compute or memory utilization percentages without quantifying the financial cost differential or data center server power impact.

---

## 3. Proposed Solution

CloudScale AI introduces a coordinated predictive and reactive optimization pipeline:

```
[ Historical Google Borg Traces / Live Telemetry ]
                        │
                        ▼
       [ Feature Extraction Pipeline (9 features) ]
                        │
                        ▼
      [ ML Inference: RandomForestRegressor v1 ]
             (Predicted CPU Demand)
                        │
                        ▼
      [ Policy B Autoscaling Recommendation Engine ]
   (Hysteresis: >87% Scale Up, <18% Scale Down + Cooldown)
                        │
                        ▼
  [ Cost & Linear Server Energy Estimation Modules ]
                        │
                        ▼
   [ FastAPI REST API + Interactive React/TS Console ]
```

1. **Short-Term Predictive Forecast**: The model forecasts near-future CPU demand based on rolling utilization trends and lag metrics.
2. **Hysteresis-Guarded Scaling**: If current utilization or predictive pressure exceeds **87.0%** ($85\% + 2\%$ hysteresis), Policy B computes an upward allocation target. If demand drops below **18.0%** ($20\% - 2\%$ hysteresis), it recommends consolidating capacity down to a safe baseline.
3. **Flapping Mitigation**: Built-in stateful cooldown tracking enforces a 2-step (or 300-second) freeze between successive scaling modifications per resource, protected by thread-safe locking (`threading.RLock`).
4. **Quantified Savings**: Before-and-after operational cost ($/hr) and estimated electrical power dissipation (Watts / Watt-hours) are computed for every recommendation.

---

## 4. Key Features

- **Google Borg Trace Telemetry Explorer**: Ingests, paginates, filters, and inspects real-world Google Borg task utilization traces with microsecond-to-second timestamp normalization.
- **Predictive Demand Forecasting**: Serves instantaneous inference using a trained 9-feature Random Forest model (`model_id: random_forest_v1`).
- **Policy B Decision Engine**: Implements bidirectional autoscaling recommendations with hysteresis boundaries, allocation constraints (minimum 1 unit, maximum 16 units), and configurable cooldown periods.
- **Simulated Energy & Power Accounting**: Computes instantaneous server power (Watts) and interval energy consumption (Watt-hours) using an industry-standard linear chassis power function ($P(u) = 100 + 150u$).
- **Offline ML Benchmark Comparator**: Surfaces rigorous offline validation metrics comparing Random Forest (Tabular), XGBoost (Tabular), and Stacked LSTM (Sequential) models across test sets derived from 65,398 Borg task pairs.
- **Interactive Operator Dashboard**: Dark-navy and cyan web UI built with React 18, Vite, TypeScript, Tailwind CSS, Lucide icons, and Recharts, featuring live telemetry graphs, KPI cards, and manual Borg trace replay.
- **Strict Data Provenance**: Explicitly demarcates historical Google Borg traces (`source = 'borg'`) from simulated runtime infrastructure (`source = 'mock'`), ensuring zero data contamination.
- **Robustness & Test Coverage**: 81 comprehensive backend assertions verifying database integrity, ML artifact validity, Policy B edge cases, and end-to-end API workflows.

---

## 5. System Architecture

### Component Architecture

```mermaid
flowchart TD
    subgraph Frontend ["Client Presentation Tier (React 18 + Vite + TypeScript)"]
        UI_Dash["Dashboard & KPIs"]
        UI_Borg["Google Borg Telemetry Explorer"]
        UI_Recs["Autoscaling Recommendations"]
        UI_ML["ML Model Analytics & Comparison"]
        API_Client["Axios HTTP Service Client"]
        UI_Dash --> API_Client
        UI_Borg --> API_Client
        UI_Recs --> API_Client
        UI_ML --> API_Client
    end

    subgraph Backend ["Application Tier (FastAPI + Pydantic v2)"]
        Auth_Router["/api/v1/auth (JWT Security)"]
        Res_Router["/api/v1/resources (Resource & Analytics APIs)"]
        Health_Router["/api/v1/health (Liveness Probe)"]
        
        PolicyB["Policy B Scaling Engine (Hysteresis & Cooldown Lock)"]
        PowerMod["Linear Power & Energy Estimation Engine"]
        ML_Service["PredictionService (Singleton Scikit-Learn Model)"]
        
        Res_Router --> PolicyB
        Res_Router --> PowerMod
        Res_Router --> ML_Service
    end

    subgraph Storage ["Data & Artifact Tier"]
        DB[("PostgreSQL 14+ Database")]
        JoblibFile[("Model Artifact: demand_predictor.joblib")]
        CSVTrace[("Google Borg Raw CSV: 65,398 rows")]
        
        Res_Router --> DB
        ML_Service --> JoblibFile
        Res_Router -.-> CSVTrace
    end

    API_Client <-->|"HTTP / REST JSON (JWT Auth)"| Res_Router
    API_Client <-->|"HTTP / REST JSON"| Auth_Router
```

### Telemetry-to-Recommendation Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Operator / Web Browser
    participant FE as React Frontend
    participant API as FastAPI Backend (/api/v1/resources)
    participant ML as PredictionService (Random Forest)
    participant Engine as Policy B Engine
    participant DB as PostgreSQL Database

    User->>FE: Select Resource & Request Analysis
    FE->>API: POST /resources/{id}/analyze
    API->>DB: Query Historical Metric Records (CPU, Mem, Lag)
    DB-->>API: Return Recent Utilization Time-Series
    
    API->>ML: predict_demand(cpu, mem, storage, network, rolling, lag, hour)
    ML-->>API: Return Predicted CPU Demand (%)
    
    API->>Engine: evaluate_policy_b(current_alloc, current_cpu, predicted_cpu)
    Note over Engine: Check Cooldown State (RLock)<br/>Apply Hysteresis (>87% Up, <18% Down)<br/>Calculate New Capacity & Linear Power P(u)
    Engine-->>API: Return Action, Target Units, Cooldown Status, Energy Wh
    
    API-->>FE: Return ScalingRecommendation JSON Payload
    FE-->>User: Render Recommendation Card, Power Delta, and Telemetry Chart
```

---

## 6. Technology Stack

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) | `0.115.11` | High-performance asynchronous Python REST API framework |
| **ASGI Web Server** | [Uvicorn](https://www.uvicorn.org/) | `0.34.0` | Production ASGI web server running FastAPI application |
| **Language Runtime** | [Python](https://www.python.org/) | `3.10+` | Core backend programming language |
| **Database** | [PostgreSQL](https://www.postgresql.org/) | `14+` | Relational storage for resources, telemetry metrics, and user accounts |
| **ORM / Migration** | [SQLAlchemy](https://www.sqlalchemy.org/) / [Psycopg 3](https://www.psycopg.org/) | `2.0.43` / `3.2.13` | Type-safe SQL toolkit and modern PostgreSQL binary driver |
| **Data Validation** | [Pydantic v2](https://docs.pydantic.dev/) | `2.13.4` | Data parsing, strict request/response schema validation, settings management |
| **Security & Auth** | [python-jose](https://github.com/mpdavis/python-jose) / [Passlib](https://passlib.readthedocs.io/) | `3.3.0` / `1.7.4` | JWT token issuance/verification and secure bcrypt password hashing |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) | `1.7.2` | Random Forest regression training, feature scaling, and inference |
| **ML Serialization** | [Joblib](https://joblib.readthedocs.io/) | `1.5.3` | High-efficiency disk serialization for trained model artifacts |
| **Numerical Computing** | [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) | `2.2.6` / `2.2.3` | Matrix operations, feature engineering, and tabular trace processing |
| **Frontend Framework** | [React](https://react.dev/) | `18.3.1` | Component-based interactive user interface |
| **Build Tooling** | [Vite](https://vitejs.dev/) | `5.4.1` | Fast development server and production bundler |
| **Frontend Language** | [TypeScript](https://www.typescriptlang.org/) | `5.5.3` | Static type safety across client interfaces and API bindings |
| **Styling** | [Tailwind CSS](https://tailwindcss.com/) | `3.4.10` | Utility-first CSS design system adhering to dark navy and cyan theme |
| **Visualizations** | [Recharts](https://recharts.org/) | `2.12.7` | Composable declarative charting library for telemetry visualization |
| **UI Iconography** | [Lucide React](https://lucide.dev/) | `0.428.0` | Accessible and lightweight iconography |

---

## 7. Machine Learning Methodology

### Active Production Model
The live API inference engine utilizes a trained `RandomForestRegressor` (`n_estimators=200`, `random_state=42`) persisted in `backend/app/ml/artifacts/demand_predictor.joblib`.

### Feature Engineering Pipeline (9 Features)
The production model consumes a feature vector engineered from sequential metric windows:

1. `cpu` — Current normalized CPU utilization ($0.0 - 100.0\%$)
2. `mem` — Current normalized Memory utilization ($0.0 - 100.0\%$)
3. `storage` — Storage consumption allocation ($0.0 - 100.0\%$)
4. `network` — Network I/O throughput rate ($0.0 - 100.0\%$)
5. `cpu_rolling_mean_3` — Rolling moving average of CPU utilization across the previous 3 intervals ($t_{-2}, t_{-1}, t_0$)
6. `memory_rolling_mean_3` — Rolling moving average of Memory utilization across the previous 3 intervals ($t_{-2}, t_{-1}, t_0$)
7. `cpu_lag_1` — Lag-1 CPU utilization observation ($t_{-1}$)
8. `memory_lag_1` — Lag-1 Memory utilization observation ($t_{-1}$)
9. `hour_of_day` — Diurnal cyclic feature extracted from the metric timestamp ($0 - 23$)

### Chronological Train/Validation Split
To prevent **look-ahead data leakage**, training and validation partitions were created using strict chronological ordering rather than random shuffling:
- **Training Set**: Earliest 80% of ordered timestamps.
- **Validation/Test Set**: Remaining 20% of ordered timestamps.
- Rolling features and lag metrics were computed strictly within sequential windows prior to split partition boundaries.

---

## 8. Dataset & Telemetry Ingestion

### Empirical Source: Google Borg 2011 Traces
The dataset utilized for training, evaluation, and telemetry replay is sampled from the publicly released **Google Borg cluster trace (clusterdata-2011-2)**:
- **Raw File**: `data/raw/google_borg_sample.csv`
- **Total Records**: 65,398 rows $\times$ 26 columns
- **Fields Extracted**: Task ID, machine ID, start/end timestamps, CPU utilization rates, memory usage, disk space allocations, and canonical memory limits.

### Telemetry Replay & Idempotency
CloudScale AI includes an automated ingestion endpoint (`POST /api/v1/resources/replay-borg`) that replays batches of historical Borg records into the PostgreSQL database:
- Extracts chronological intervals and associates them with unique Borg resource records.
- Enforces database idempotency via timestamp matching and unique constraints to prevent duplicate insertions across repeated executions.
- Ingestion telemetry is assigned `source = 'borg'`, ensuring strict isolation from runtime mock resources (`source = 'mock'`).

---

## 9. Autoscaling Strategy (Policy B)

CloudScale AI implements **Policy B**, an enhanced autoscaling heuristic designed to mitigate the shortcomings of naive threshold alerting.

### Core Mathematical Rules

$$\text{Scale Up Trigger: } \max(u_{\text{cpu}}, u_{\text{pred}}) > 87.0\% \quad (85\% + 2\% \text{ hysteresis})$$

$$\text{Scale Down Trigger: } \max(u_{\text{cpu}}, u_{\text{pred}}) < 18.0\% \quad (20\% - 2\% \text{ hysteresis})$$

$$\text{Capacity Bounds: } C_{\min} = 1, \quad C_{\max} = 16 \text{ instances/units}$$

### Anti-Flapping Cooldown Mechanism
To prevent destructive oscillations under noisy or rapidly fluctuating workloads:
- **Discrete Step Cooldown**: Enforces a minimum count of evaluation cycles (`POLICY_B_COOLDOWN_STEPS = 2`) between scaling actions.
- **Wall-Clock Cooldown**: Enforces a minimum elapsed time window (`POLICY_B_COOLDOWN_SECONDS = 300`) before an allocated resource can be resized again.
- **Thread Safety**: Process-level concurrency is guarded using a Python re-entrant lock (`threading.RLock`) within the scaling engine, ensuring thread-safe state mutations during concurrent API requests.

> **Note on Cloud Execution**: Policy B generates **simulated allocation recommendations**. It calculates the ideal instance count and verifies scaling safety, but does not mutate real third-party cloud infrastructure (e.g., AWS EC2 or GCP GCE) unless explicitly wired to an authenticated provider SDK.

---

## 10. Cost & Energy Optimization Models

### 1. Simulated Financial Cost Model
Operational expenditure is computed using standard cloud on-demand hourly pricing benchmarks:

$$\text{Hourly Cost (\$/hr)} = C \times \text{Rate}_{\text{instance}}$$

Where:
- $C$ is the number of active compute units (cores/instances).
- $\text{Rate}_{\text{instance}}$ is the hourly billing rate (baseline: \$0.0400/hr per unit).
- Cost Delta represents the direct hourly expenditure difference between the current allocation and Policy B's recommended capacity.

### 2. Linear Server Power & Energy Model
Data center server electrical power dissipation is calculated using an established linear power function:

$$P(u) = P_{\text{idle}} + (P_{\text{peak}} - P_{\text{idle}}) \times u_{\text{norm}} = 100\text{ W} + 150\text{ W} \times u_{\text{norm}}$$

Where:
- $P_{\text{idle}} = 100\text{ Watts}$ (chassis base idle draw).
- $P_{\text{peak}} = 250\text{ Watts}$ (server maximum load draw).
- $u_{\text{norm}} \in [0.01, 1.0]$ is the normalized per-core CPU load fraction:
  
  $$u_{\text{norm}} = \min\left(1.0, \max\left(0.01, \frac{u_{\text{cpu}} / 100.0}{C}\right)\right)$$

Energy consumption across an evaluation window ($\Delta t = 300\text{ seconds}$ / 5 minutes) is calculated as:

$$E(\text{Wh}) = P(u) \times \left(\frac{\Delta t_{\text{seconds}}}{3600}\right) = P(u) \times \left(\frac{300}{3600}\right) = \frac{P(u)}{12}$$

Power savings are derived from the delta between the estimated energy consumption of the current over-provisioned allocation versus the optimized recommended capacity.

---

## 11. Model Evaluation & Benchmark Results

During offline research benchmarking (**Phase 4 Model Comparison**), three distinct model architectures were evaluated on identical historical partitions of the Google Borg workload dataset.

### Benchmark Performance Summary

| Architecture | Input Representation | Test Set Size | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Coefficient of Determination ($R^2$) | Single-Sample Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** (Active Production) | 18 Engineered Features (Tabular) | 6,540 pairs | **0.000615** | **0.001899** | **0.9071** | **0.0293 ms** |
| **XGBoost** | 18 Engineered Features (Tabular) | 6,540 pairs | 0.000877 | 0.002290 | 0.8650 | 0.0115 ms |
| **Stacked LSTM** | $5 \times 7$ Temporal Sequence Tensors | 1,560 sequences | 0.000417 | 0.000547 | 0.9838 | 0.2369 ms |

### Important Scientific Notes on Evaluation
1. **$R^2$ is the Coefficient of Determination**: The reported $R^2$ scores ($0.9071$ for Random Forest, $0.9838$ for LSTM) represent the proportion of variance in CPU demand explained by the model features. **$R^2$ is not percentage classification accuracy**.
2. **Variance in Test Sample Counts**: 
   - Tabular models (Random Forest and XGBoost) evaluated across **6,540** point-in-time test pairs.
   - The Sequential Stacked LSTM was evaluated across **1,560** continuous 5-step sequence tensors. The sample count difference is attributable to strict continuity filtering ($L=5$, interval gap $\le 1800\text{ s}$) required to construct continuous temporal sequences without synthetic interpolation.
3. **Model Selection Rationale**: While the Stacked LSTM demonstrated superior $R^2$ in offline evaluation, **Random Forest** was selected for active production deployment due to its ultra-low inference latency ($29.3\,\mu\text{s}$), deterministic memory profile, and zero external C++ runtime dependencies.

---

## 12. Project Directory Structure

```
D:\PBL
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── deps.py                  # Database session and auth dependency injection
│   │   │   └── v1/
│   │   │       ├── auth.py              # Login and JWT bearer token issuance
│   │   │       └── resources.py         # Telemetry, analytics, and scaling endpoints
│   │   ├── core/
│   │   │   ├── config.py                # Pydantic v2 BaseSettings system configuration
│   │   │   └── security.py              # Password hashing and JWT encoding/decoding
│   │   ├── db/
│   │   │   ├── base.py                  # Declarative base model imports
│   │   │   └── session.py               # SQLAlchemy engine and sessionmaker
│   │   ├── ml/
│   │   │   ├── artifacts/
│   │   │   │   └── demand_predictor.joblib  # Active Random Forest serialized model
│   │   │   └── predictor.py             # PredictionService class and feature transformer
│   │   ├── models/
│   │   │   ├── resource.py              # Resource and ResourceMetric SQLAlchemy models
│   │   │   └── user.py                  # User authentication model
│   │   ├── schemas/
│   │   │   ├── resource.py              # Pydantic schemas for requests and responses
│   │   │   ├── token.py                 # JWT token response schema
│   │   │   └── user.py                  # User request/response schemas
│   │   ├── services/
│   │   │   └── autoscaling.py           # Policy B scaling engine & power calculations
│   │   └── main.py                      # FastAPI application entrypoint and middleware
│   ├── tests/                           # 4 Automated test suites (81 assertions)
│   │   ├── test_stage1_borg.py
│   │   ├── test_stage2_ml.py
│   │   ├── test_stage3_policy_b.py
│   │   └── test_stage4_e2e_verification.py
│   ├── .env.example                     # Environment configuration template
│   └── requirements.txt                 # Backend Python package dependencies
├── frontend/
│   ├── src/
│   │   ├── components/                  # Navigation bar, sidebar, and layout primitives
│   │   ├── services/
│   │   │   ├── api.ts                   # Axios client instance and interceptors
│   │   │   └── resourceService.ts       # Typed API bindings for all backend routes
│   │   ├── types/
│   │   │   └── index.ts                 # TypeScript interfaces for API models
│   │   ├── views/
│   │   │   ├── DashboardView.tsx        # System KPIs, power savings, and status
│   │   │   ├── ResourcesView.tsx        # Google Borg trace explorer and replay UI
│   │   │   ├── RecommendationsView.tsx  # Policy B scaling recommendations
│   │   │   └── ModelAnalyticsView.tsx   # ML metrics, comparison table, and retraining
│   │   ├── App.tsx                      # Root application router and view switcher
│   │   └── main.tsx                     # React DOM entrypoint
│   ├── package.json                     # Frontend npm package dependencies
│   ├── tsconfig.json                    # TypeScript compiler options
│   └── vite.config.ts                   # Vite bundler configuration
├── data/
│   ├── raw/
│   │   └── google_borg_sample.csv       # 65,398-row sample of Google Borg traces
│   └── models/
│       └── phase4_comparison/           # Offline benchmark metrics and comparative reports
│           ├── model_comparison_metrics.json
│           └── model_comparison_report.md
├── docs/
│   └── architecture.md                  # Comprehensive architectural specification
└── README.md                            # Complete project documentation
```

---

## 13. Prerequisites

Ensure the following runtimes and services are installed on your host system:

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: Version `3.10` or higher (`python --version`)
- **Node.js**: Version `18.0.0` or higher (`node --version`)
- **Package Manager**: `npm` (v9+) or `pnpm`
- **Database**: [PostgreSQL](https://www.postgresql.org/) version `14` or higher running locally or accessible via network

---

## 14. Installation & Local Setup

### Step 1: Clone the Repository
Open a terminal (e.g., Windows PowerShell or bash) and navigate to your workspace:
```powershell
git clone <repository-url> D:\PBL
Set-Location -Path "D:\PBL"
```

### Step 2: Configure PostgreSQL Database
Ensure PostgreSQL is running, then create the database:
```powershell
# Using psql command line (adjust user if necessary)
psql -U postgres -c "CREATE DATABASE cloudscale_ai;"
```

### Step 3: Backend Setup & Virtual Environment
1. Navigate to the `backend` directory:
   ```powershell
   Set-Location -Path "D:\PBL\backend"
   ```
2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install required Python packages:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Configure your environment file:
   ```powershell
   Copy-Item .env.example .env
   # Edit .env to verify your PostgreSQL database credentials
   ```
5. Initialize database tables and seed test data:
   ```powershell
   python -m app.db.init_db
   ```
6. Start the FastAPI development server:
   ```powershell
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The backend service is now live at `http://localhost:8000` (API Docs: `http://localhost:8000/docs`).

### Step 4: Frontend Setup
1. Open a new PowerShell terminal and navigate to the `frontend` directory:
   ```powershell
   Set-Location -Path "D:\PBL\frontend"
   ```
2. Install frontend dependencies:
   ```powershell
   npm install
   ```
3. Start the Vite development server:
   ```powershell
   npm run dev
   ```
4. Access the web interface at `http://localhost:5173`.

---

## 15. Configuration & Environment Variables

Backend settings are managed using Pydantic Settings in `backend/app/core/config.py`. Configure these variables in `backend/.env`:

| Variable | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `PROJECT_NAME` | String | `CloudScale AI` | Human-readable project title |
| `API_V1_STR` | String | `/api/v1` | URL prefix for version 1 API endpoints |
| `DATABASE_URL` | String | `postgresql://postgres:postgres@localhost:5432/cloudscale_ai` | SQLAlchemy PostgreSQL connection URI |
| `SECRET_KEY` | String | *Required in production* | Secret key for signing HMAC-SHA256 JWT tokens |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Integer | `43200` (30 days) | JWT bearer token lifetime in minutes |
| `FIRST_SUPERUSER_EMAIL` | String | `admin@cloudscale.ai` | Default seed administrator account email |
| `FIRST_SUPERUSER_PASSWORD` | String | `admin123` | Default seed administrator account password |
| `SCALE_UP_THRESHOLD` | Float | `85.0` | Base CPU percentage threshold for scale-up alerts |
| `SCALE_DOWN_THRESHOLD` | Float | `20.0` | Base CPU percentage threshold for scale-down alerts |
| `POLICY_B_HYSTERESIS` | Float | `2.0` | Hysteresis guard band applied to thresholds ($\pm 2.0\%$) |
| `POLICY_B_COOLDOWN_STEPS` | Integer | `2` | Number of evaluation cycles required between scaling events |
| `POLICY_B_COOLDOWN_SECONDS`| Integer | `300` | Minimum wall-clock seconds required between scaling events |
| `BORG_DATASET_PATH` | String | `../data/raw/google_borg_sample.csv` | Relative or absolute path to the raw Borg sample CSV |

---

## 16. REST API Documentation

All API routes are served under the `/api/v1` prefix. Interactive Swagger/OpenAPI documentation is available at `/docs`.

### Primary Endpoints

| Method | Endpoint | Auth Required | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | No | System health check and liveness probe |
| `POST` | `/api/v1/auth/login` | No | Authenticates credentials and returns a JWT bearer access token |
| `GET` | `/api/v1/resources` | Yes (Bearer) | Lists all cloud and Borg infrastructure resources |
| `GET` | `/api/v1/resources/{id}` | Yes (Bearer) | Retrieves details and current allocation for a specific resource |
| `POST` | `/api/v1/resources/{id}/analyze` | Yes (Bearer) | Executes Policy B scaling analysis and power estimation for a resource |
| `GET` | `/api/v1/resources/{id}/metrics` | Yes (Bearer) | Retrieves time-series metric records for a specific resource |
| `GET` | `/api/v1/resources/model/info` | Yes (Bearer) | Retrieves metadata, features, and status of the active ML model |
| `GET` | `/api/v1/resources/model/comparison` | Yes (Bearer) | Retrieves offline validation metrics comparing RF, XGBoost, and LSTM |
| `POST` | `/api/v1/resources/model/train` | Superuser | Triggers retraining of the production demand prediction model |
| `GET` | `/api/v1/resources/borg/summary` | Yes (Bearer) | Returns dynamic record counts and status of Borg traces in the database |
| `GET` | `/api/v1/resources/borg/metrics` | Yes (Bearer) | Paginated telemetry records from historical Google Borg cluster traces |
| `POST` | `/api/v1/resources/replay-borg` | Superuser | Ingests 100 historical Google Borg task records (idempotent) |
| `GET` | `/api/v1/resources/energy/summary` | Yes (Bearer) | Aggregated system power consumption and energy savings summary |
| `POST` | `/api/v1/resources/cooldown/reset` | Superuser | Resets the Policy B in-memory cooldown state across all resources |

### Example Request & Response: Resource Scaling Analysis

**Request:**
```http
POST /api/v1/resources/1/analyze HTTP/1.1
Host: localhost:8000
Authorization: Bearer <jwt-token>
Content-Type: application/json
```

**Response (`200 OK`):**
```json
{
  "resource_id": 1,
  "resource_name": "borg-task-instance-1",
  "current_allocation": 4,
  "recommended_allocation": 2,
  "action": "SCALE_DOWN",
  "reason": "Predicted demand (14.2%) is below scale-down boundary (18.0%). Cooldown expired.",
  "confidence_score": 0.907,
  "current_cost": 0.1600,
  "recommended_cost": 0.0800,
  "potential_savings": 0.0800,
  "savings_percentage": 50.0,
  "current_power_watts": 122.50,
  "recommended_power_watts": 110.65,
  "current_energy_wh": 10.21,
  "recommended_energy_wh": 9.22,
  "energy_savings_wh": 0.99,
  "energy_savings_percentage": 9.67,
  "in_cooldown": false
}
```

---

## 17. Verification, Testing & Code Quality

CloudScale AI features four comprehensive test suites containing **81 verified assertions** covering data ingestion, ML models, Policy B logic, and API workflows.

### Running Backend Test Suites
Execute the test suites inside `backend` with the virtual environment active:

```powershell
Set-Location -Path "D:\PBL\backend"

# Suite 1: Google Borg Telemetry & Ingestion (8 assertions)
.\venv\Scripts\python.exe tests/test_stage1_borg.py

# Suite 2: ML Model Artifact & Prediction Pipeline (26 assertions)
.\venv\Scripts\python.exe tests/test_stage2_ml.py

# Suite 3: Policy B Hysteresis & Cooldown Verification (28 assertions)
.\venv\Scripts\python.exe tests/test_stage3_policy_b.py

# Suite 4: End-to-End API Integration Verification (19 assertions)
.\venv\Scripts\python.exe tests/test_stage4_e2e_verification.py
```

### Running Frontend Type Checks & Production Build
Validate TypeScript typings and build the production bundle:

```powershell
Set-Location -Path "D:\PBL\frontend"

# Compile TypeScript and build production bundle
npm run build
```

---

## 18. User Interface & Dashboards

The user interface follows a modern dark-navy and cyan design system:

- **Dashboard View**: High-level overview displaying aggregate cloud resource counts, active recommendations, estimated financial cost reductions, and aggregate server power savings (Watts and Watt-hours).
- **Google Borg Trace Explorer (`ResourcesView`)**: Paginated telemetry table showing empirical Borg task records, CPU/memory distributions, and manual trace replay controls with duplicate-prevention notices.
- **Recommendations Console (`RecommendationsView`)**: Real-time evaluation view presenting Policy B scaling recommendations, current vs. target instance counts, cooldown status indicators, and power consumption deltas.
- **Model Analytics & Comparison (`ModelAnalyticsView`)**: Interactive model evaluation center displaying live Random Forest metrics ($R^2$, MAE, RMSE), comparative benchmark tables (RF vs. XGBoost vs. LSTM), and retrain actions.

---

## 19. System Limitations & Assumptions

For academic transparency, users and evaluators should consider the following design characteristics:

1. **Simulation Environment**: Autoscaling decisions and power measurements are **simulated models**. The system computes analytical allocations and power values; it does not issue live hypervisor calls to AWS/GCP or interface with physical IPMI/wattmeter hardware.
2. **Linear Power Approximation**: The energy model adopts an established linear chassis relationship ($P(u) = 100 + 150u$). It does not capture complex dynamic frequency scaling (DVFS), ambient CRAC cooling overhead (PUE), or multi-socket NUMA bus contentions.
3. **Trace Resampling**: Google Borg cluster traces are sampled at discrete intervals (typically 300 seconds); microsecond-level scheduling bursts occurring within sub-interval windows are smoothed out during preprocessing.
4. **Single-Process Cooldown State**: In the current implementation, Policy B cooldown timers are tracked via in-memory thread-safe dictionaries (`threading.RLock`). In a horizontally scaled multi-worker production setup, cooldown states should be externalized to a distributed store such as Redis.

---

## 20. Contributing Guidelines

Contributions to CloudScale AI are welcome. Please follow these conventions:

1. **Fork the Repository**: Create a dedicated feature branch (`git checkout -b feature/improvement-name`).
2. **Follow Coding Standards**:
   - Python: Follow PEP 8 style guidelines with strict type hints.
   - TypeScript: Maintain strict typing without `any` overrides.
3. **Preserve Test Integrity**: Ensure all 81 automated backend assertions pass and `npm run build` succeeds without compiler errors before submitting pull requests.
4. **Submit Pull Request**: Open a PR with a clear, technical description of the proposed changes.

---

## 21. License

This project is currently distributed as an academic research prototype. The official open-source license has not yet been specified. Please consult the project maintainers for permissions regarding commercial distribution or derivative use.

---

## 22. Authors & Acknowledgments

- **CloudScale AI Research Team**: Project Based Learning (PBL) Engineering Cohort.
- **Google LLC**: For releasing the Google Borg 2011 cluster dataset (`clusterdata-2011-2`) to the academic and systems research community.
