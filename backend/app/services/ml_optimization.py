"""ML-based optimization engine with baseline fallback.

Produces an optimization analysis backed by the validated
``RandomForestRegressor`` demand prediction model. If the model is
unavailable or prediction cannot be completed (missing artifact,
insufficient data, heterogeneous data sources, or feature preparation failure),
it transparently falls back to the deterministic :class:`BaselineRuleEngine`.

The response always includes detailed prediction provenance:
- model_id & model_type
- prediction_target
- input_data_source ('google_borg', 'mock', etc.)
- is_fallback (True/False)
- sufficient_history (True/False)
"""

from __future__ import annotations

import logging
from app.ml.predictor import ModelNotAvailableError, PredictionService
from app.models.metric import ResourceMetric
from app.models.resource import CloudResource
from app.schemas.optimization import (
    AnalysisSource,
    CurrentUtilization,
    EstimatedImpact,
    OptimizationAnalysis,
    PredictionProvenance,
    UtilizationStatus,
)
from app.services.optimization import BaseOptimizationEngine, BaselineRuleEngine

logger = logging.getLogger(__name__)

# Minimum consecutive historical metric samples required for reliable 9-feature extraction
MIN_METRICS_FOR_PREDICTION = 3


class MLOptimizationEngine(BaseOptimizationEngine):
    """Optimization engine combining validated ML demand prediction with baseline fallback."""

    def __init__(
        self,
        prediction_service: PredictionService | None = None,
        baseline_engine: BaselineRuleEngine | None = None,
    ) -> None:
        self.prediction_service = prediction_service or PredictionService()
        self.baseline_engine = baseline_engine or BaselineRuleEngine()

    def analyze_resource(
        self,
        resource: CloudResource,
        metrics: list[ResourceMetric] | None = None,
    ) -> OptimizationAnalysis:
        """Produce an optimization analysis, preferring ML-backed prediction.

        Args:
            resource: The cloud resource being analyzed.
            metrics: Optional chronological historical metric samples.

        Returns:
            An ``OptimizationAnalysis`` whose ``analysis_source`` is
            ``ml_prediction`` when a validated prediction succeeded, otherwise
            ``baseline`` with complete provenance.
        """
        # Validate data sufficiency and source continuity
        if metrics and len(metrics) >= MIN_METRICS_FOR_PREDICTION:
            # Check source homogeneity to prevent mixing Borg and mock data in one feature vector
            sources = set(m.source for m in metrics if getattr(m, "source", None))
            if len(sources) <= 1:
                try:
                    predicted_cpu = self.prediction_service.predict_cpu_utilization(metrics)
                    provenance_data = self.prediction_service.get_provenance(metrics)
                    provenance = PredictionProvenance(**provenance_data)
                    return self._build_ml_analysis(
                        resource=resource,
                        predicted_cpu=predicted_cpu,
                        provenance=provenance,
                    )
                except (ModelNotAvailableError, ValueError, Exception) as exc:
                    logger.warning(
                        "ML inference failed for resource %s: %s; falling back to baseline rules.",
                        resource.id,
                        exc,
                    )

        # Baseline fallback path
        baseline_res = self.baseline_engine.analyze_resource(resource)
        
        sample_count = len(metrics) if metrics else 0
        data_sources = set(m.source for m in metrics) if metrics else {resource.provider_type}
        source_str = ",".join(sorted(data_sources)) if data_sources else "resource_snapshot"

        fallback_provenance = PredictionProvenance(
            model_id="baseline_rule_engine",
            model_type="BaselineRuleEngine",
            prediction_target="current_utilization_thresholds",
            input_data_source=source_str,
            is_fallback=True,
            sufficient_history=sample_count >= MIN_METRICS_FOR_PREDICTION,
            sample_count_used=sample_count,
            model_version="1.0.0",
        )

        baseline_res.provenance = fallback_provenance
        baseline_res.model_used = "Rule-Based Engine"
        return baseline_res

    def _build_ml_analysis(
        self,
        resource: CloudResource,
        predicted_cpu: float,
        provenance: PredictionProvenance,
    ) -> OptimizationAnalysis:
        """Build a structured optimization analysis from an ML prediction."""
        cpu = resource.cpu_utilization
        memory = resource.memory_utilization
        storage = resource.storage_utilization
        network = resource.network_utilization
        cost = resource.estimated_cost

        current_util = CurrentUtilization(
            cpu=cpu,
            memory=memory,
            storage=storage,
            network=network,
        )

        if predicted_cpu > 85.0:
            status = UtilizationStatus.OVERLOADED
            recommendation = (
                f"Resource '{resource.name}' is projected to reach high load "
                f"(predicted CPU: {predicted_cpu:.1f}%). "
                "Consider upgrading capacity before degradation occurs."
            )
            suggested_action = "Increase capacity or add auto-scaling replicas based on ML demand forecast."
            impact = EstimatedImpact(
                cost_change_percentage=25.0,
                estimated_monthly_savings=0.0,
                performance_impact="Prevents predicted latency spikes and service degradation.",
            )
            confidence = 0.92
            decision_reason = (
                f"ML model ({provenance.model_id}) predicted CPU={predicted_cpu:.1f}pct (>85). "
                "Scale-up to prevent overload."
            )
        elif predicted_cpu < 20.0 and memory < 20.0:
            status = UtilizationStatus.UNDERUTILIZED
            estimated_savings = round(cost * 0.40, 2)
            recommendation = (
                f"Resource '{resource.name}' is projected to remain underutilized "
                f"(predicted CPU: {predicted_cpu:.1f}%). "
                "Downsizing or consolidating workloads can achieve significant cost savings."
            )
            suggested_action = "Downsize to a smaller instance tier based on ML demand forecast."
            impact = EstimatedImpact(
                cost_change_percentage=-40.0,
                estimated_monthly_savings=estimated_savings,
                performance_impact="Minimal performance impact expected for low-demand workload.",
            )
            confidence = 0.90
            decision_reason = (
                f"ML model ({provenance.model_id}) predicted CPU={predicted_cpu:.1f}pct and "
                f"Mem={memory:.1f}pct (both <20). Downsize to cut idle cost."
            )
        else:
            status = UtilizationStatus.OPTIMAL
            recommendation = (
                f"Resource '{resource.name}' is projected to operate within optimal capacity "
                f"(predicted CPU: {predicted_cpu:.1f}%)."
            )
            suggested_action = "Maintain current resource provisioning based on ML demand forecast."
            impact = EstimatedImpact(
                cost_change_percentage=0.0,
                estimated_monthly_savings=0.0,
                performance_impact="Stable performance maintained with current capacity.",
            )
            confidence = 0.93
            decision_reason = (
                f"ML model ({provenance.model_id}) predicted CPU={predicted_cpu:.1f}pct "
                "within optimal range 20-85. No scaling needed."
            )

        return OptimizationAnalysis(
            resource_id=resource.id,
            resource_name=resource.name,
            status=status,
            recommendation=recommendation,
            current_utilization=current_util,
            suggested_action=suggested_action,
            estimated_impact=impact,
            confidence_score=confidence,
            analysis_source=AnalysisSource.ML_PREDICTION,
            predicted_cpu_utilization=round(predicted_cpu, 2),
            model_used=provenance.model_type,
            provenance=provenance,
            decision_reason=decision_reason,
        )
