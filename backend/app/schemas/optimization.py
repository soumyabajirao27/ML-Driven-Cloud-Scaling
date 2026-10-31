"""Optimization AI analysis schemas (Pydantic v2).

Defines structured response models for CloudScale AI optimization engine,
including Phase 5 Policy B autoscaling recommendations, cooldown/hysteresis
safeguards, and simulated cost/power estimates.
"""

from enum import Enum
from pydantic import Field

from app.schemas.base import ORMModel


class UtilizationStatus(str, Enum):
    """Classification status for resource utilization."""

    UNDERUTILIZED = "UNDERUTILIZED"
    OPTIMAL = "OPTIMAL"
    OVERLOADED = "OVERLOADED"


class ScalingAction(str, Enum):
    """Available scaling recommendations and actions."""

    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    MAINTAIN = "MAINTAIN"
    DEFERRED_COOLDOWN = "DEFERRED_COOLDOWN"
    NO_ACTION_INSUFFICIENT_DATA = "NO_ACTION_INSUFFICIENT_DATA"


class CurrentUtilization(ORMModel):
    """Summary of utilization metrics for analysis."""

    cpu: float = Field(..., description="CPU utilization percentage")
    memory: float = Field(..., description="Memory utilization percentage")
    storage: float = Field(..., description="Storage utilization percentage")
    network: float = Field(..., description="Network utilization percentage")


class EstimatedImpact(ORMModel):
    """Estimated financial and performance impact of optimization."""

    cost_change_percentage: float = Field(..., description="Estimated percentage change in cost (negative = savings)")
    estimated_monthly_savings: float = Field(..., description="Estimated monthly savings in USD or local currency units")
    performance_impact: str = Field(..., description="Description of expected performance impact")


class AnalysisSource(str, Enum):
    """Indicates which engine produced the optimization analysis."""

    ML_PREDICTION = "ml_prediction"
    BASELINE = "baseline"


class PredictionProvenance(ORMModel):
    """Metadata regarding ML inference origin, configuration, and data sufficiency."""

    model_id: str = Field(..., description="Identifier of the model used (e.g. 'random_forest_v1')")
    model_type: str = Field(..., description="Architecture type: 'RandomForestRegressor' or 'BaselineRuleEngine'")
    prediction_target: str = Field(..., description="Quantity predicted: 'next_sample_cpu_utilization_pct'")
    input_data_source: str = Field(..., description="Data origin: 'google_borg', 'mock', or 'resource_snapshot'")
    is_fallback: bool = Field(..., description="True if analysis fell back to baseline heuristics")
    sufficient_history: bool = Field(..., description="True if enough historical timesteps existed for inference")
    sample_count_used: int = Field(default=0, description="Number of historical metric samples evaluated")
    model_version: str = Field(default="1.0.0", description="Version identifier of the active model")


class CostEstimate(ORMModel):
    """Normalized cost estimates for Policy A baseline vs Policy B recommendation."""

    baseline_cost: float = Field(..., description="Normalized baseline cost for current allocation")
    recommended_cost: float = Field(..., description="Normalized recommended cost under policy allocation")
    simulated_cost_difference: float = Field(..., description="Difference in cost (recommended - baseline)")
    simulated_cost_percentage_change: float = Field(..., description="Percentage change in cost")
    unit: str = Field(default="cost_units", description="Unit of cost measurement")


class EnergyEstimate(ORMModel):
    """Simulated power and energy consumption derived from server linear power model."""

    baseline_power_w: float = Field(..., description="Estimated baseline power consumption in Watts")
    recommended_power_w: float = Field(..., description="Estimated power under recommended allocation in Watts")
    baseline_energy_wh: float = Field(..., description="Estimated baseline energy consumption in Watt-hours")
    recommended_energy_wh: float = Field(..., description="Estimated recommended energy consumption in Watt-hours")
    simulated_energy_percentage_change: float = Field(..., description="Simulated percentage change in energy")
    unit: str = Field(default="Watt-hours (Wh)", description="Unit of energy measurement")
    power_model: str = Field(default="Linear (100W Idle - 250W Max)", description="Server power model specification")


class SystemEnergySummary(ORMModel):
    """Aggregate simulated power and energy across all connected cloud resources."""

    total_resources: int = Field(..., description="Total count of evaluated resources")
    baseline_power_w: float = Field(..., description="Total baseline power draw across all resources (Watts)")
    recommended_power_w: float = Field(..., description="Total recommended power draw across all resources (Watts)")
    baseline_energy_wh: float = Field(..., description="Total baseline energy consumption across all resources (Watt-hours)")
    recommended_energy_wh: float = Field(..., description="Total recommended energy consumption across all resources (Watt-hours)")
    energy_savings_wh: float = Field(..., description="Total simulated energy savings (Wh) = Baseline - Recommended")
    energy_savings_kwh: float = Field(..., description="Total simulated energy savings in kiloWatt-hours (kWh)")
    energy_savings_percentage: float = Field(..., description="Percentage change in energy (positive = reduction / savings)")
    evaluation_interval_sec: float = Field(default=300.0, description="Evaluation period duration in seconds")
    power_model: str = Field(default="Linear (100W Idle - 250W Max)", description="Server power model specification")
    is_simulated: bool = Field(default=True, description="Indicates simulated energy estimation")


class OptimizationAnalysis(ORMModel):
    """Structured AI optimization result for a cloud resource."""

    resource_id: int = Field(..., description="ID of the analyzed resource")
    resource_name: str = Field(..., description="Name of the analyzed resource")
    status: UtilizationStatus = Field(..., description="Resource utilization status classification")
    recommendation: str = Field(..., description="Actionable optimization recommendation")
    current_utilization: CurrentUtilization = Field(..., description="Current resource utilization breakdown")
    suggested_action: str = Field(..., description="Specific recommended scaling or tuning action")
    action_type: ScalingAction = Field(
        default=ScalingAction.MAINTAIN,
        description="Standardized scaling action type (SCALE_UP, SCALE_DOWN, MAINTAIN, DEFERRED_COOLDOWN)",
    )
    estimated_impact: EstimatedImpact = Field(..., description="Estimated cost and performance impact")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of analysis (0.0 to 1.0)")
    analysis_source: AnalysisSource = Field(
        default=AnalysisSource.BASELINE,
        description="Engine that produced this analysis (ml_prediction or baseline)",
    )
    predicted_cpu_utilization: float | None = Field(
        default=None,
        description="Predicted future CPU utilization percentage (only when ML prediction is used)",
    )
    model_used: str = Field(
        default="Rule-Based Engine",
        description="Model or engine used: 'RandomForestRegressor' or 'Rule-Based Engine'",
    )
    policy_id: str = Field(
        default="Policy B (Proactive ML)",
        description="Identifier of the active autoscaling policy",
    )
    provenance: PredictionProvenance | None = Field(
        default=None,
        description="Detailed ML inference provenance and telemetry integrity metadata",
    )
    safeguard_status: str = Field(
        default="Normal",
        description="Safeguard status applied (e.g., Cooldown Active, Bounds Enforced, Normal)",
    )
    cooldown_remaining_steps: int = Field(
        default=0,
        description="Number of cooldown steps remaining before another scaling action can trigger",
    )
    cooldown_remaining_seconds: float = Field(
        default=0.0,
        description="Estimated seconds remaining in cooldown window",
    )
    hysteresis_active: bool = Field(
        default=False,
        description="True if the recommendation was damped or influenced by the hysteresis deadband",
    )
    allocated_capacity: float = Field(
        default=1.0,
        description="Recommended capacity allocation in capacity units",
    )
    decision_reason: str = Field(
        default="",
        description="Transparent explanation of the optimization policy decision",
    )
    cost_estimate: CostEstimate | None = Field(
        default=None,
        description="Simulated normalized cost analysis comparing baseline to recommendation",
    )
    energy_estimate: EnergyEstimate | None = Field(
        default=None,
        description="Simulated power and energy analysis comparing baseline to recommendation",
    )
    is_simulated: bool = Field(
        default=True,
        description="Indicates that policy recommendations are simulated and no real hardware was altered",
    )
    action_recorded: bool = Field(
        default=False,
        description="Indicates whether this recommendation was recorded into stateful policy history",
    )


class CooldownResetRequest(ORMModel):
    """Payload to reset cooldown state for testing or operational overrides."""

    resource_id: int | None = Field(default=None, description="Target resource ID, or null to reset all")


__all__ = [
    "UtilizationStatus",
    "ScalingAction",
    "CurrentUtilization",
    "EstimatedImpact",
    "OptimizationAnalysis",
    "AnalysisSource",
    "PredictionProvenance",
    "CostEstimate",
    "EnergyEstimate",
    "SystemEnergySummary",
    "CooldownResetRequest",
]
