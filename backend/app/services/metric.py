"""Resource metric business service layer.

Coordinates historical metric CRUD operations and ML model training through
the existing repository contract and ML modules.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.exceptions import ResourceNotFoundException
from app.ml.trainer import TrainingMetrics, train_demand_model
from app.models.metric import ResourceMetric
from app.models.resource import CloudResource
from app.repositories.metric import MetricRepository
from app.schemas.metric import MetricCreate, TrainingResult


class MetricService:
    """Business logic for historical resource metrics and ML training."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.metric_repository = MetricRepository(db)

    def create_metric(
        self,
        resource: CloudResource,
        metric_data: MetricCreate,
    ) -> ResourceMetric:
        """Record a new historical metric observation for a cloud resource."""
        timestamp = metric_data.timestamp or datetime.now(timezone.utc)

        metric = ResourceMetric(
            resource_id=resource.id,
            timestamp=timestamp,
            cpu_utilization=metric_data.cpu_utilization,
            memory_utilization=metric_data.memory_utilization,
            storage_utilization=metric_data.storage_utilization,
            network_utilization=metric_data.network_utilization,
            source=metric_data.source or "mock",
        )
        return self.metric_repository.create(metric)

    def list_metrics(
        self,
        resource_id: int,
        limit: int | None = None,
        source: str | None = None,
    ) -> list[ResourceMetric]:
        """Return chronological metric samples for a resource."""
        return self.metric_repository.get_metrics_for_resource(
            resource_id=resource_id,
            limit=limit,
            source=source,
        )

    def get_metrics_for_analysis(
        self,
        resource_id: int,
        limit: int | None = None,
    ) -> list[ResourceMetric]:
        """Fetch chronological metrics suitable for ML prediction/analysis."""
        return self.metric_repository.get_metrics_for_resource(
            resource_id=resource_id,
            limit=limit,
        )

    def train_model(self) -> TrainingResult:
        """Train (or retrain) the demand prediction model on all metrics.

        Aggregates metric samples across all resources for a larger
        training corpus. Returns actual computed metrics (MAE, RMSE, R²)
        or an ``insufficient_data`` result when not enough samples exist.

        Raises:
            ResourceNotFoundException: If no metric samples exist at all.
        """
        all_metrics = self.metric_repository.get_all()
        if not all_metrics:
            raise ResourceNotFoundException(message="No metric samples available for training")

        try:
            training: TrainingMetrics = train_demand_model(all_metrics)
        except ValueError as exc:
            return TrainingResult(
                status="insufficient_data",
                samples_used=len(all_metrics),
                mae=None,
                rmse=None,
                r2=None,
                model_version=None,
                message=str(exc),
            )

        return TrainingResult(
            status="success",
            samples_used=training.samples_used,
            mae=round(training.mae, 4),
            rmse=round(training.rmse, 4),
            r2=round(training.r2, 4),
            model_version=training.model_version,
            message="Model trained successfully with RandomForestRegressor",
        )
