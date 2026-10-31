"""Resource metric Pydantic schemas (Pydantic v2).

Request and response schemas for historical resource utilization metrics.
"""

from datetime import datetime

from pydantic import Field

from app.schemas.base import ORMModel, UTCModel


class MetricCreate(ORMModel):
    """Payload for creating a new historical metric observation."""

    timestamp: datetime | None = Field(default=None, description="Observation timestamp (defaults to now)")
    cpu_utilization: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage")
    memory_utilization: float = Field(..., ge=0.0, le=100.0, description="Memory usage percentage")
    storage_utilization: float = Field(default=0.0, ge=0.0, le=100.0, description="Storage usage percentage")
    network_utilization: float = Field(default=0.0, ge=0.0, le=100.0, description="Network usage percentage")
    source: str = Field(default="mock", max_length=50, description="Workload telemetry source ('mock', 'google_borg')")


class MetricResponse(UTCModel):
    """Public representation of a stored metric sample."""

    id: int
    resource_id: int
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    storage_utilization: float
    network_utilization: float
    source: str = "mock"
    resource_name: str | None = None


class TrainingResult(ORMModel):
    """Structured result from ML model training."""

    status: str = Field(..., description="Training outcome (e.g. 'success', 'insufficient_data')")
    samples_used: int = Field(..., description="Number of training samples")
    mae: float | None = Field(default=None, description="Mean Absolute Error on validation set")
    rmse: float | None = Field(default=None, description="Root Mean Squared Error on validation set")
    r2: float | None = Field(default=None, description="R-squared score on validation set")
    model_version: str | None = Field(default=None, description="Version identifier of the trained model")
    message: str = Field(..., description="Human-readable summary")


class ModelInfoResponse(ORMModel):
    """Structured metadata about the active ML model."""

    model_id: str
    model_type: str
    model_version: str
    target: str
    expected_features_count: int
    status: str
    is_available: bool


__all__ = ["MetricCreate", "MetricResponse", "TrainingResult", "ModelInfoResponse"]
