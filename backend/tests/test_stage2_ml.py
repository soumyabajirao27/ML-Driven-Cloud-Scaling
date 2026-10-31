"""Stage 2 automated tests -- ML model evaluation, selection, and integration.

Validates:
  - PredictionService loads the correct 9-feature RandomForest artifact
  - PredictionService fails gracefully with a missing artifact
  - PredictionService produces valid numeric output (0-100 clamped)
  - PredictionService.get_provenance() returns correct metadata
  - MLOptimizationEngine returns provenance with is_fallback=False when ML succeeds
  - MLOptimizationEngine returns provenance with is_fallback=True when insufficient metrics
  - MLOptimizationEngine rejects mixed-source metrics (Borg + mock)
  - OptimizationAnalysis.model_used reflects actual model type, not hardcoded string
  - Stage 1 ingestion records remain intact
  - PredictionProvenance schema roundtrips correctly

Run with:
    Set-Location d:\PBL\backend
    .\venv\Scripts\python.exe tests\test_stage2_ml.py
"""

from __future__ import annotations

import os
import sys
import datetime

# -- path setup ----------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

# -- counts ---------------------------------------------------------------
passed = 0
failed = 0
errors: list[str] = []


def ok(label: str) -> None:
    global passed
    passed += 1
    print(f"  [OK] {label}")


def fail(label: str, reason: str) -> None:
    global failed
    failed += 1
    errors.append(f"{label}: {reason}")
    print(f"  [FAIL] {label} -- {reason}")


# -------------------------------------------------------------------------
# Lightweight stub objects that duck-type as SQLAlchemy model instances
# without triggering instrumented attribute descriptors.
# -------------------------------------------------------------------------
class FakeMetric:
    """Mimics ResourceMetric for feature extraction and provenance tests."""

    def __init__(
        self,
        cpu: float,
        mem: float,
        storage: float,
        network: float,
        ts: datetime.datetime,
        source: str = "test",
    ):
        self.cpu_utilization = cpu
        self.memory_utilization = mem
        self.storage_utilization = storage
        self.network_utilization = network
        self.timestamp = ts
        self.source = source


class FakeResource:
    """Mimics CloudResource for MLOptimizationEngine tests."""

    def __init__(self):
        self.id = 999
        self.name = "test-resource"
        self.cpu_utilization = 35.0
        self.memory_utilization = 40.0
        self.storage_utilization = 50.0
        self.network_utilization = 10.0
        self.estimated_cost = 100.0
        self.provider_type = "test"


# Build reusable test fixtures
now = datetime.datetime.utcnow()
fake_metrics = [
    FakeMetric(30.0 + i * 2, 40.0 + i, 50.0, 10.0, now + datetime.timedelta(minutes=i * 5), "test")
    for i in range(5)
]
fake_resource = FakeResource()


# =========================================================================
# T1  PredictionService loads 9-feature RF artifact
# =========================================================================
print("\n=== T1: PredictionService loads model artifact ===")
try:
    from app.ml.predictor import PredictionService

    svc = PredictionService()
    model = svc._load_model()
    n = getattr(model, "n_features_in_", None)
    if n == 9:
        ok(f"Model loaded with n_features_in_={n}")
    else:
        fail("Feature count", f"Expected 9, got {n}")
except Exception as exc:
    fail("Load model", str(exc))

# =========================================================================
# T2  PredictionService fails gracefully with missing artifact
# =========================================================================
print("\n=== T2: PredictionService fails on missing artifact ===")
try:
    from app.ml.predictor import ModelNotAvailableError

    bad_svc = PredictionService(model_path="/nonexistent/model.joblib")
    try:
        bad_svc._load_model()
        fail("Missing artifact", "Expected ModelNotAvailableError")
    except ModelNotAvailableError:
        ok("ModelNotAvailableError raised for missing artifact")
except Exception as exc:
    fail("Missing artifact test", str(exc))

# =========================================================================
# T3  PredictionService produces valid numeric output (0-100 clamped)
# =========================================================================
print("\n=== T3: PredictionService numeric prediction ===")
try:
    svc = PredictionService()
    prediction = svc.predict_cpu_utilization(fake_metrics)
    if isinstance(prediction, float) and 0.0 <= prediction <= 100.0:
        ok(f"Prediction = {prediction:.2f} (valid float in 0-100)")
    else:
        fail("Prediction range", f"Got {prediction}")
except Exception as exc:
    fail("Numeric prediction", str(exc))

# =========================================================================
# T4  PredictionService raises ValueError on empty metrics
# =========================================================================
print("\n=== T4: PredictionService rejects empty metrics ===")
try:
    svc = PredictionService()
    try:
        svc.predict_cpu_utilization([])
        fail("Empty metrics", "Expected ValueError")
    except ValueError:
        ok("ValueError raised for empty metrics list")
except Exception as exc:
    fail("Empty metrics test", str(exc))

# =========================================================================
# T5  PredictionService.get_provenance() returns correct metadata
# =========================================================================
print("\n=== T5: PredictionService provenance metadata ===")
try:
    svc = PredictionService()
    prov = svc.get_provenance(fake_metrics)

    checks = [
        ("model_id", prov.get("model_id") == "random_forest_v1"),
        ("model_type", prov.get("model_type") == "RandomForestRegressor"),
        ("prediction_target", prov.get("prediction_target") == "next_sample_cpu_utilization_pct"),
        ("is_fallback", prov.get("is_fallback") is False),
        ("sufficient_history", prov.get("sufficient_history") is True),
        ("sample_count_used", prov.get("sample_count_used") == 5),
        ("model_version", prov.get("model_version") == "1.0.0"),
        ("input_data_source", prov.get("input_data_source") == "test"),
    ]
    for name, cond in checks:
        if cond:
            ok(f"provenance.{name} correct")
        else:
            fail(f"provenance.{name}", f"got {prov.get(name)}")
except Exception as exc:
    fail("Provenance metadata", str(exc))

# =========================================================================
# T6  PredictionProvenance schema roundtrips correctly
# =========================================================================
print("\n=== T6: PredictionProvenance schema roundtrip ===")
try:
    from app.schemas.optimization import PredictionProvenance

    pp = PredictionProvenance(
        model_id="random_forest_v1",
        model_type="RandomForestRegressor",
        prediction_target="next_sample_cpu_utilization_pct",
        input_data_source="google_borg",
        is_fallback=False,
        sufficient_history=True,
        sample_count_used=29,
        model_version="1.0.0",
    )
    d = pp.model_dump()
    if d["model_id"] == "random_forest_v1" and d["is_fallback"] is False and d["sample_count_used"] == 29:
        ok("PredictionProvenance roundtrip OK")
    else:
        fail("PredictionProvenance roundtrip", f"Unexpected dict: {d}")
except Exception as exc:
    fail("PredictionProvenance schema", str(exc))

# =========================================================================
# T7  MLOptimizationEngine returns is_fallback=False on valid ML path
# =========================================================================
print("\n=== T7: MLOptimizationEngine ML prediction path ===")
try:
    from app.services.ml_optimization import MLOptimizationEngine

    engine = MLOptimizationEngine()
    analysis = engine.analyze_resource(fake_resource, fake_metrics)

    if analysis.provenance is not None:
        if analysis.provenance.is_fallback is False:
            ok("ML path: is_fallback=False")
        else:
            fail("ML path", f"is_fallback={analysis.provenance.is_fallback}")

        if analysis.analysis_source.value == "ml_prediction":
            ok("ML path: analysis_source=ml_prediction")
        else:
            fail("ML path", f"analysis_source={analysis.analysis_source}")

        if analysis.model_used == "RandomForestRegressor":
            ok("ML path: model_used=RandomForestRegressor")
        else:
            fail("ML path model_used", f"got '{analysis.model_used}'")

        if analysis.predicted_cpu_utilization is not None and 0 <= analysis.predicted_cpu_utilization <= 100:
            ok(f"ML path: predicted_cpu={analysis.predicted_cpu_utilization:.2f}")
        else:
            fail("ML path predicted_cpu", f"got {analysis.predicted_cpu_utilization}")
    else:
        fail("ML path", "provenance is None")
except Exception as exc:
    fail("MLOptimizationEngine ML path", str(exc))

# =========================================================================
# T8  MLOptimizationEngine returns is_fallback=True with insufficient data
# =========================================================================
print("\n=== T8: MLOptimizationEngine baseline fallback (insufficient data) ===")
try:
    engine = MLOptimizationEngine()
    # Provide only 1 metric (below MIN_METRICS_FOR_PREDICTION=3)
    analysis_bl = engine.analyze_resource(fake_resource, [fake_metrics[0]])

    if analysis_bl.provenance is not None:
        if analysis_bl.provenance.is_fallback is True:
            ok("Baseline fallback: is_fallback=True")
        else:
            fail("Baseline fallback", f"is_fallback={analysis_bl.provenance.is_fallback}")

        if analysis_bl.analysis_source.value == "baseline":
            ok("Baseline fallback: analysis_source=baseline")
        else:
            fail("Baseline fallback", f"analysis_source={analysis_bl.analysis_source}")
    else:
        fail("Baseline fallback", "provenance is None")
except Exception as exc:
    fail("MLOptimizationEngine fallback", str(exc))

# =========================================================================
# T9  MLOptimizationEngine falls back on mixed sources
# =========================================================================
print("\n=== T9: MLOptimizationEngine mixed-source rejection ===")
try:
    mixed_metrics = [
        FakeMetric(30.0 + i * 2, 40.0 + i, 50.0, 10.0,
                   now + datetime.timedelta(minutes=i * 5),
                   "google_borg" if i < 2 else "mock")
        for i in range(4)
    ]

    engine = MLOptimizationEngine()
    analysis_mx = engine.analyze_resource(fake_resource, mixed_metrics)

    if analysis_mx.provenance is not None and analysis_mx.provenance.is_fallback is True:
        ok("Mixed source: correctly fell back to baseline")
    else:
        fallback = analysis_mx.provenance.is_fallback if analysis_mx.provenance else "no provenance"
        fail("Mixed source", f"Expected fallback, got is_fallback={fallback}")
except Exception as exc:
    fail("Mixed source rejection", str(exc))

# =========================================================================
# T10  OptimizationAnalysis schema contains provenance field
# =========================================================================
print("\n=== T10: OptimizationAnalysis schema has provenance ===")
try:
    from app.schemas.optimization import OptimizationAnalysis
    fields = OptimizationAnalysis.model_fields
    if "provenance" in fields:
        ok("provenance field exists in OptimizationAnalysis")
    else:
        fail("OptimizationAnalysis schema", "provenance field missing")

    if "model_used" in fields:
        ok("model_used field exists in OptimizationAnalysis")
    else:
        fail("OptimizationAnalysis schema", "model_used field missing")

    if "analysis_source" in fields:
        ok("analysis_source field exists in OptimizationAnalysis")
    else:
        fail("OptimizationAnalysis schema", "analysis_source field missing")
except Exception as exc:
    fail("OptimizationAnalysis schema check", str(exc))

# =========================================================================
# T11  Stage 1 resource metric model has source field
# =========================================================================
print("\n=== T11: Stage 1 compatibility -- source field on ResourceMetric ===")
try:
    from app.models.metric import ResourceMetric as RM
    from sqlalchemy import inspect as sa_inspect

    mapper = sa_inspect(RM)
    col_names = [c.key for c in mapper.columns]
    if "source" in col_names:
        ok("ResourceMetric model has 'source' column")
    else:
        fail("ResourceMetric source", f"Columns: {col_names}")
except Exception as exc:
    fail("Stage 1 source field", str(exc))

# =========================================================================
# T12  Config includes ML inference settings
# =========================================================================
print("\n=== T12: Config ML settings ===")
try:
    from app.core.config import settings

    if hasattr(settings, "ML_MODEL_TYPE") and settings.ML_MODEL_TYPE == "random_forest":
        ok("ML_MODEL_TYPE = 'random_forest'")
    else:
        fail("ML_MODEL_TYPE", f"got {getattr(settings, 'ML_MODEL_TYPE', 'MISSING')}")

    if hasattr(settings, "ML_CONFIDENCE_THRESHOLD"):
        ok(f"ML_CONFIDENCE_THRESHOLD = {settings.ML_CONFIDENCE_THRESHOLD}")
    else:
        fail("ML_CONFIDENCE_THRESHOLD", "missing from settings")
except Exception as exc:
    fail("Config ML settings", str(exc))

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 60)
total = passed + failed
print(f"Stage 2 Tests: {passed}/{total} passed, {failed} failed")
if errors:
    print("\nFailures:")
    for e in errors:
        print(f"  - {e}")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
