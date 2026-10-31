"""ResourceMetric ORM model.

Stores time-series historical utilization metrics for cloud resources.
Inherits shared ``BaseModel`` columns (``id``, ``created_at``, ``updated_at``).
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class ResourceMetric(BaseModel):
    """Historical metric sample for a cloud resource.

    Attributes:
        resource_id: Foreign key referencing ``cloud_resources.id``.
        timestamp: Timezone-aware timestamp of the metric observation.
        cpu_utilization: Percentage CPU usage (0.0 to 100.0).
        memory_utilization: Percentage Memory usage (0.0 to 100.0).
        storage_utilization: Percentage Storage usage (0.0 to 100.0).
        network_utilization: Percentage Network usage (0.0 to 100.0).
        source: Workload telemetry source ('mock', 'google_borg', etc.).
    """

    __tablename__ = "resource_metrics"

    resource_id: Mapped[int] = mapped_column(
        ForeignKey("cloud_resources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
    cpu_utilization: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    memory_utilization: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    storage_utilization: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    network_utilization: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    source: Mapped[str] = mapped_column(
        String(50),
        default="mock",
        nullable=False,
        index=True,
    )

    __table_args__ = (
        Index("ix_resource_metrics_res_time", "resource_id", "timestamp"),
    )
