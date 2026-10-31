# CloudScale AI: Architecture & System Flows

This document details the concrete, implemented system architecture, data ingestion and inference pipeline, and optimization decision flows of the **CloudScale AI** platform.

---

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend Layer (React 18 + Vite + TypeScript)"]
        UI["Tailwind + Lucide UI Dashboard"]
        Views["Views: Recommendations, Predictions, Monitoring, FinOps"]
        Service["optimizationService & resourceService"]
        UI --> Views --> Service
    end

    subgraph API ["Application Server (FastAPI v1)"]
        Router["/api/v1/resources Router"]
        AuthMiddleware["JWT Bearer Authentication & RBAC"]
        ResService["ResourceService & BorgReplayService"]
        Engine["PolicyBOptimizationEngine"]
        Fallback["BaselineRuleEngine"]
        StateMgr["ScalingStateManager (Thread-safe Singleton)"]
        MLPredictor["PredictionService (Joblib RandomForest)"]
        
        Router --> AuthMiddleware --> ResService
        ResService --> Engine
        Engine -->|Data Incomplete / Corrupted| Fallback
        Engine --> MLPredictor
        Engine <--> StateMgr
    end

    subgraph Storage ["Persistence Layer (PostgreSQL 16)"]
        UsersTbl["users (Account & Superuser Flags)"]
        ResourcesTbl["cloud_resources (Provider, Cost, Capacity)"]
        MetricsTbl["resource_metrics (CPU, Mem, Source Provenance)"]
    end

    subgraph Dataset ["Workload Telemetry"]
        BorgCSV["Google Borg Traces (borg_temporal_pairs.csv)"]
        ModelArtifact["Trained ML Artifact (demand_predictor.joblib)"]
    end

    Service <-->|REST JSON / Bearer Token| Router
    ResService <-->|SQLAlchemy ORM + Psycopg 3| Storage
    BorgCSV -.->|Bounded Ingestion max 100 intervals| ResService
    ModelArtifact -.->|Model Deserialization| MLPredictor
```

---

## 2. Telemetry Ingestion and Prediction Data Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / Evaluator
    participant API as FastAPI (/api/v1/resources)
    participant Replay as BorgReplayService
    participant DB as PostgreSQL (resource_metrics)
    participant ML as PredictionService
    participant Engine as PolicyBOptimizationEngine
    participant State as ScalingStateManager

    User->>API: POST /replay-borg (max_intervals=100)
    API->>Replay: Validate & Chronologically Sort CSV Samples
    Replay->>DB: Ingest unique metric rows (source='google_borg')
    DB-->>API: Imported count & ignored duplicate stats
    API-->>User: Ingestion Response

    User->>API: POST /{resource_id}/analyze
    API->>DB: Fetch chronological metrics (filtered by resource_id)
    DB-->>API: list[ResourceMetric]

    alt Metrics >= 3 and Single Source
        API->>ML: prepare_single_prediction_features(metrics)
        Note over ML: Builds 9-dim vector:<br/>[cpu, mem, stor, net, rolling3_cpu, rolling3_mem, lag1_cpu, lag1_mem, hour]
        ML->>ML: predict_cpu_utilization()
        ML-->>Engine: predicted_cpu (clamped 0-100%) + Provenance
    else Metrics < 3 or Mixed Sources
        API->>Engine: Fallback to BaselineRuleEngine (is_fallback=True)
    end

    Engine->>State: check_cooldown(resource_id)
    State-->>Engine: (is_in_cooldown, remaining_steps, remaining_seconds)

    alt In Cooldown
        Engine->>Engine: Action: DEFERRED_COOLDOWN (allocation held steady)
    else Predicted CPU > 87% (>85% + 2% Hysteresis)
        Engine->>Engine: Action: SCALE_UP (Proactive capacity doubling)
        Engine->>State: record_action(resource_id, "SCALE_UP")
    else Predicted CPU < 18% (<20% - 2% Hysteresis)
        Engine->>Engine: Action: SCALE_DOWN (Proactive 50% right-sizing)
        Engine->>State: record_action(resource_id, "SCALE_DOWN")
    else Neutral Band (18% <= CPU <= 87%)
        Engine->>Engine: Action: MAINTAIN (Allocated capacity held)
        Engine->>State: record_action(resource_id, "MAINTAIN")
    end

    Engine->>Engine: calculate_cost_estimate() & calculate_energy_estimate()
    Engine-->>API: OptimizationAnalysis (with provenance, safeguards, estimates)
    API-->>User: JSON Response (is_simulated=True)
```

---

## 3. Policy B Optimization & Safeguard Decision Flow

```mermaid
flowchart TD
    Start["POST /resources/{id}/analyze"] --> LoadMetrics["Retrieve Historical Metrics"]
    LoadMetrics --> CheckSufficiency{"Length >= 3 Samples?"}
    
    CheckSufficiency -- No --> BaselineFallback["Activate BaselineRuleEngine\n(Set is_fallback=True)"]
    CheckSufficiency -- Yes --> CheckHomogeneity{"Single Consistent Source?\n(e.g. only 'google_borg')"}
    
    CheckHomogeneity -- No --> BaselineFallback
    CheckHomogeneity -- Yes --> MLInference["Run RandomForestRegressor\nFeature extraction (9 features)\nOutput: predicted_cpu (clamped 0-100%)"]
    
    MLInference --> CooldownCheck{"Cooldown Safeguard Active?\n(steps > 0 or seconds remaining)"}
    
    CooldownCheck -- Yes --> ActionDeferred["Action: DEFERRED_COOLDOWN\nSafeguard: Cooldown Active\nAllocation: Maintain current\nStep cooldown timer"]
    CooldownCheck -- No --> EvalHysteresis{"Evaluate Hysteresis Deadband\n(85% ± 2%, 20% ± 2%)"}
    
    EvalHysteresis -- "predicted_cpu > 87.0%" --> ActionScaleUp["Action: SCALE_UP\nProactive capacity scaling\nInitiate 2-step cooldown"]
    EvalHysteresis -- "predicted_cpu < 18.0%" --> ActionScaleDown["Action: SCALE_DOWN\nProactive capacity right-sizing\nInitiate 2-step cooldown"]
    EvalHysteresis -- "18.0% <= predicted_cpu <= 87.0%" --> ActionMaintain["Action: MAINTAIN\nNeutral Deadband (Stable)\nHold allocation steady"]
    
    BaselineFallback --> ComputeEstimates
    ActionDeferred --> ComputeEstimates
    ActionScaleUp --> ComputeEstimates
    ActionScaleDown --> ComputeEstimates
    ActionMaintain --> ComputeEstimates
    
    ComputeEstimates["Compute Estimates:\n1. Cost Model: Alloc_cpu * 1.0 + Mem * 0.5\n2. Power Model: 100W Idle + Util * (250W - 100W)\n3. Energy: Power * (300s / 3600s) Wh"]
    
    ComputeEstimates --> FinalPayload["Construct OptimizationAnalysis\n- Flag is_simulated=True\n- Attach provenance\n- Return HTTP 200 OK"]
```

