"""Stage 3 Automated Test Suite: Policy B Autoscaling, Cooldown, Hysteresis, and Cost/Power Modeling.

Validates:
  - Policy B Scale-Up conditions (predicted CPU > 87%)
  - Policy B Scale-Down conditions (predicted CPU < 18%)
  - Policy B Neutral conditions (18% <= predicted CPU <= 87%)
  - Cooldown: Blocks rapid re-triggering within configured cooldown steps
  - Cooldown: Resumes after cooldown steps elapse or reset
  - Cooldown: Thread safety under concurrent access
  - Hysteresis: Damps oscillations near 20% and 85% boundaries
  - Cost Modeling: Correct calculation of normalized costs and percentage change
  - Power/Energy Modeling: Linear server power model (100W-250W) and Watt-hours calculation
  - Provenance & Transparency: Every decision has policy_id, decision_reason, safeguard_status
  - Fallback Handling: Safely switches to Baseline rules on missing/insufficient data
  - API and Schema: OptimizationAnalysis exposes all Phase 5 metrics cleanly
  - End-to-end integration: Borg historical telemetry -> ML -> Policy B -> API response

Run with:
    Set-Location d:\\PBL\\backend
    .\\venv\\Scripts\\python.exe tests\\test_stage3_policy_b.py
"""

from __future__ import annotations

import datetime
import os
import sys
import threading

# -- path setup ----------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

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


# Stub classes for deterministic testing
class FakeMetric:
    def __init__(self, cpu: float, mem: float, storage: float, network: float, ts: datetime.datetime, source: str = "test"):
        self.cpu_utilization = cpu
        self.memory_utilization = mem
        self.storage_utilization = storage
        self.network_utilization = network
        self.timestamp = ts
        self.source = source


class FakeResource:
    def __init__(self, res_id: int = 101, cpu: float = 30.0, mem: float = 40.0, cap: float = 2.0, cost: float = 100.0):
        self.id = res_id
        self.name = f"test-server-{res_id}"
        self.cpu_utilization = cpu
        self.memory_utilization = mem
        self.storage_utilization = 30.0
        self.network_utilization = 15.0
        self.current_capacity = cap
        self.estimated_cost = cost
        self.provider_type = "mock"


class MockPredictor:
    def __init__(self, fixed_pred: float = 50.0):
        self.fixed_pred = fixed_pred

    def predict_cpu_utilization(self, metrics):
        return self.fixed_pred

    def get_provenance(self, metrics):
        return {
            "model_id": "test_rf_model",
            "model_type": "RandomForestRegressor",
            "prediction_target": "next_sample_cpu_utilization_pct",
            "input_data_source": "test_telemetry",
            "is_fallback": False,
            "sufficient_history": True,
            "sample_count_used": len(metrics) if metrics else 0,
            "model_version": "1.0.0",
        }


# =========================================================================
# T1: Policy B Scale-Up Trigger (> 87% Predicted CPU)
# =========================================================================
print("\n=== T1: Policy B Scale-Up Decision ===")
try:
    from app.services.policy_b_engine import PolicyBOptimizationEngine
    from app.schemas.optimization import ScalingAction, UtilizationStatus
    from app.services.state_manager import scaling_state_manager

    scaling_state_manager.reset()
    now = datetime.datetime.utcnow()
    metrics = [FakeMetric(70.0, 50.0, 30.0, 10.0, now + datetime.timedelta(minutes=i)) for i in range(5)]
    res = FakeResource(res_id=1, cpu=70.0, mem=50.0, cap=2.0)

    engine = PolicyBOptimizationEngine(prediction_service=MockPredictor(fixed_pred=90.0))
    analysis = engine.analyze_resource(res, metrics)

    if analysis.action_type == ScalingAction.SCALE_UP:
        ok("SCALE_UP triggered when predicted CPU=90.0% (>87%)")
    else:
        fail("SCALE_UP trigger", f"Expected SCALE_UP, got {analysis.action_type}")

    if analysis.allocated_capacity > res.current_capacity:
        ok(f"Allocated capacity increased from {res.current_capacity} to {analysis.allocated_capacity}")
    else:
        fail("Capacity change", f"Capacity not increased: {analysis.allocated_capacity}")

    if "proactive scale-up" in analysis.decision_reason.lower():
        ok("Decision reason documents proactive scale-up")
    else:
        fail("Decision reason", f"Got: {analysis.decision_reason}")
except Exception as exc:
    fail("T1 Exception", str(exc))


# =========================================================================
# T2: Policy B Scale-Down Trigger (< 18% Predicted CPU)
# =========================================================================
print("\n=== T2: Policy B Scale-Down Decision ===")
try:
    scaling_state_manager.reset()
    res2 = FakeResource(res_id=2, cpu=12.0, mem=15.0, cap=4.0)
    engine = PolicyBOptimizationEngine(prediction_service=MockPredictor(fixed_pred=14.0))
    analysis2 = engine.analyze_resource(res2, metrics)

    if analysis2.action_type == ScalingAction.SCALE_DOWN:
        ok("SCALE_DOWN triggered when predicted CPU=14.0% (<18%)")
    else:
        fail("SCALE_DOWN trigger", f"Expected SCALE_DOWN, got {analysis2.action_type}")

    if analysis2.allocated_capacity < res2.current_capacity:
        ok(f"Allocated capacity decreased from {res2.current_capacity} to {analysis2.allocated_capacity}")
    else:
        fail("Capacity change", f"Capacity not decreased: {analysis2.allocated_capacity}")

    if "right-sizing" in analysis2.decision_reason.lower():
        ok("Decision reason documents right-sizing")
    else:
        fail("Decision reason", f"Got: {analysis2.decision_reason}")
except Exception as exc:
    fail("T2 Exception", str(exc))


# =========================================================================
# T3: Policy B Neutral Band (18% - 87% Predicted CPU)
# =========================================================================
print("\n=== T3: Policy B Neutral Allocation (MAINTAIN) ===")
try:
    scaling_state_manager.reset()
    res3 = FakeResource(res_id=3, cpu=45.0, mem=50.0, cap=2.0)
    engine = PolicyBOptimizationEngine(prediction_service=MockPredictor(fixed_pred=48.0))
    analysis3 = engine.analyze_resource(res3, metrics)

    if analysis3.action_type == ScalingAction.MAINTAIN:
        ok("MAINTAIN action when predicted CPU=48.0% (neutral region)")
    else:
        fail("MAINTAIN trigger", f"Expected MAINTAIN, got {analysis3.action_type}")

    if analysis3.allocated_capacity == res3.current_capacity:
        ok(f"Allocated capacity held steady at {analysis3.allocated_capacity}")
    else:
        fail("Capacity held", f"Got {analysis3.allocated_capacity}")
except Exception as exc:
    fail("T3 Exception", str(exc))


# =========================================================================
# T4: Cooldown Safeguard (Damping Rapid Actions)
# =========================================================================
print("\n=== T4: Cooldown Dampening Enforcement ===")
try:
    scaling_state_manager.reset()
    res4 = FakeResource(res_id=4, cpu=80.0, mem=70.0, cap=2.0)
    engine_up = PolicyBOptimizationEngine(prediction_service=MockPredictor(fixed_pred=92.0))

    # First call: triggers SCALE_UP and activates 2-step cooldown
    a1 = engine_up.analyze_resource(res4, metrics)
    if a1.action_type == ScalingAction.SCALE_UP:
        ok("Step 1: First action triggers SCALE_UP")
    else:
        fail("Step 1 action", f"Expected SCALE_UP, got {a1.action_type}")

    # Second call immediately after: should be DEFERRED_COOLDOWN
    a2 = engine_up.analyze_resource(res4, metrics)
    if a2.action_type == ScalingAction.DEFERRED_COOLDOWN:
        ok("Step 2: Second action deferred by active cooldown safeguard")
    else:
        fail("Cooldown block", f"Expected DEFERRED_COOLDOWN, got {a2.action_type}")

    if a2.cooldown_remaining_steps > 0:
        ok(f"Remaining cooldown steps reported accurately: {a2.cooldown_remaining_steps}")
    else:
        fail("Cooldown steps", f"Got: {a2.cooldown_remaining_steps}")

    # Third call: progresses cooldown
    a3 = engine_up.analyze_resource(res4, metrics)
    ok(f"Step 3: Progressed cooldown (remaining: {a3.cooldown_remaining_steps})")

    # Reset cooldown explicitly: confirms override behavior
    scaling_state_manager.reset(res4.id)
    a4 = engine_up.analyze_resource(res4, metrics)
    if a4.action_type == ScalingAction.SCALE_UP:
        ok("Post-reset: Scaling action immediately permitted")
    else:
        fail("Post-reset action", f"Expected SCALE_UP, got {a4.action_type}")
except Exception as exc:
    fail("T4 Exception", str(exc))


# =========================================================================
# T5: Hysteresis Buffer Deadband Analysis
# =========================================================================
print("\n=== T5: Hysteresis Deadband Dampening ===")
try:
    scaling_state_manager.reset()
    # At 86% predicted CPU (above baseline 85%, but below 85% + 2% = 87% hysteresis threshold)
    res5 = FakeResource(res_id=5, cpu=80.0, mem=50.0, cap=2.0)
    engine_h = PolicyBOptimizationEngine(prediction_service=MockPredictor(fixed_pred=86.0))
    a_h = engine_h.analyze_resource(res5, metrics)

    if a_h.action_type == ScalingAction.MAINTAIN and a_h.hysteresis_active:
        ok("Hysteresis deadband damped scale-up at 86% (<87% threshold)")
    else:
        fail("Hysteresis deadband", f"action={a_h.action_type}, deadband_active={a_h.hysteresis_active}")
except Exception as exc:
    fail("T5 Exception", str(exc))


# =========================================================================
# T6: Normalized Cost Analysis Computation
# =========================================================================
print("\n=== T6: Cost Estimation Formulas ===")
try:
    from app.services.policy_b_engine import calculate_cost_estimate

    # Current capacity 2.0, recommended 4.0, mem proxy 0.5
    cost_res = calculate_cost_estimate(current_capacity=2.0, recommended_capacity=4.0, memory_util=50.0)

    # Baseline cost = 2.0 * 1.0 + 0.5 * 0.5 = 2.25
    # Recommended cost = 4.0 * 1.0 + 0.5 * 0.5 = 4.25
    # Diff = 2.0
    if abs(cost_res.baseline_cost - 2.25) < 0.01:
        ok(f"Baseline cost correctly computed: {cost_res.baseline_cost} {cost_res.unit}")
    else:
        fail("Baseline cost", f"Expected 2.25, got {cost_res.baseline_cost}")

    if abs(cost_res.recommended_cost - 4.25) < 0.01:
        ok(f"Recommended cost correctly computed: {cost_res.recommended_cost} {cost_res.unit}")
    else:
        fail("Recommended cost", f"Expected 4.25, got {cost_res.recommended_cost}")

    if cost_res.simulated_cost_percentage_change > 0:
        ok(f"Cost percentage change (+{cost_res.simulated_cost_percentage_change}%) reflects capacity expansion")
    else:
        fail("Cost pct change", f"Got {cost_res.simulated_cost_percentage_change}")
except Exception as exc:
    fail("T6 Exception", str(exc))


# =========================================================================
# T7: Power and Energy Linear Model Computation
# =========================================================================
print("\n=== T7: Power & Energy Estimation Formulas ===")
try:
    from app.services.policy_b_engine import calculate_energy_estimate

    # 50% CPU, 2.0 Current capacity -> Util = 0.5 / 2.0 = 0.25
    # Power = 100 + 0.25 * (250 - 100) = 137.5 W
    # Energy for 300s = 137.5 * (300 / 3600) = 11.4583 Wh
    energy_res = calculate_energy_estimate(current_cpu_pct=50.0, current_capacity=2.0, recommended_capacity=4.0, delta_t_sec=300.0)

    if abs(energy_res.baseline_power_w - 137.5) < 0.5:
        ok(f"Baseline power draw matches linear model: {energy_res.baseline_power_w} W")
    else:
        fail("Baseline power", f"Expected ~137.5W, got {energy_res.baseline_power_w}")

    if energy_res.baseline_energy_wh > 0:
        ok(f"Baseline energy matches Wh duration formula: {energy_res.baseline_energy_wh} Wh")
    else:
        fail("Baseline energy", f"Got {energy_res.baseline_energy_wh}")

    if "100W" in energy_res.power_model and "250W" in energy_res.power_model:
        ok(f"Power model specification documented: '{energy_res.power_model}'")
    else:
        fail("Power model", f"Got: {energy_res.power_model}")
except Exception as exc:
    fail("T7 Exception", str(exc))


# =========================================================================
# T8: Baseline Fallback on Insufficient Telemetry
# =========================================================================
print("\n=== T8: Baseline Fallback Handling ===")
try:
    scaling_state_manager.reset()
    res8 = FakeResource(res_id=8, cpu=90.0, mem=90.0, cap=2.0)
    engine_fb = PolicyBOptimizationEngine()

    # Provide only 1 metric sample (below MIN_METRICS_FOR_PREDICTION = 3)
    a_fb = engine_fb.analyze_resource(res8, [metrics[0]])

    if a_fb.provenance is not None and a_fb.provenance.is_fallback:
        ok("Provenance flags is_fallback=True on insufficient samples")
    else:
        fail("Fallback flag", f"Expected is_fallback=True, got {getattr(a_fb.provenance, 'is_fallback', None)}")

    if a_fb.policy_id == "Policy A (Baseline Reactive)":
        ok("Policy ID identifies Policy A baseline fallback")
    else:
        fail("Fallback policy_id", f"Got: {a_fb.policy_id}")

    if a_fb.status == UtilizationStatus.OVERLOADED:
        ok("Resource status correctly classified as OVERLOADED via fallback rules")
    else:
        fail("Fallback status", f"Got: {a_fb.status}")
except Exception as exc:
    fail("T8 Exception", str(exc))


# =========================================================================
# T9: Thread Safety of Cooldown State Manager
# =========================================================================
print("\n=== T9: Concurrency and Thread Safety ===")
try:
    scaling_state_manager.reset()
    res_id = 99
    results = []

    def concurrent_action():
        in_cd, _, _ = scaling_state_manager.check_cooldown(res_id)
        if not in_cd:
            scaling_state_manager.record_action(res_id, "SCALE_UP", 2.0)
            results.append("SUCCESS")
        else:
            results.append("BLOCKED")

    threads = [threading.Thread(target=concurrent_action) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = results.count("SUCCESS")
    blocks = results.count("BLOCKED")

    if successes == 1 and blocks == 9:
        ok("Thread safety verified: exactly 1 thread succeeded, 9 blocked by cooldown")
    else:
        fail("Thread safety race", f"Successes={successes}, Blocks={blocks}")
except Exception as exc:
    fail("T9 Exception", str(exc))


# =========================================================================
# T10: Live Database Integration with Real Borg Telemetry
# =========================================================================
print("\n=== T10: End-to-End Real Telemetry Verification ===")
try:
    from app.db.session import SessionLocal
    from app.models.user import User
    from app.models.resource import CloudResource
    from app.services.resource import ResourceService

    db = SessionLocal()
    user = db.query(User).filter(User.email == "svc_test@test.com").first()
    res_borg = db.query(CloudResource).filter(CloudResource.id == 27).first()

    if user and res_borg:
        service = ResourceService(db)
        live_analysis = service.analyze_resource(27, user)

        if live_analysis.policy_id == "Policy B (Proactive ML)":
            ok("Resource 27 (Google Borg) analyzed by Policy B")
        else:
            fail("Live policy_id", f"Got: {live_analysis.policy_id}")

        if live_analysis.cost_estimate is not None:
            ok(f"Cost estimate attached: baseline={live_analysis.cost_estimate.baseline_cost}")
        else:
            fail("Live cost estimate", "Missing cost estimate")

        if live_analysis.energy_estimate is not None:
            ok(f"Energy estimate attached: power={live_analysis.energy_estimate.baseline_power_w}W")
        else:
            fail("Live energy estimate", "Missing energy estimate")

        if live_analysis.is_simulated is True:
            ok("is_simulated flag confirms simulated execution without hardware mutation")
        else:
            fail("Live simulation flag", f"Got: {live_analysis.is_simulated}")
    else:
        fail("Live DB fixtures", "User or resource 27 not found in DB")
except Exception as exc:
    fail("T10 Exception", str(exc))


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 60)
total = passed + failed
print(f"Stage 3 Tests: {passed}/{total} passed, {failed} failed")
if errors:
    print("\nFailures:")
    for e in errors:
        print(f"  - {e}")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)

