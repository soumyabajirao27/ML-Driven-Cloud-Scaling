"""Cloud Resource Pydantic schemas (Pydantic v2).

Request and response schemas for CloudScale AI cloud resources/workloads.
Includes numeric validation for utilization percentages and capacities.
"""

from pydantic import Field

from app.schemas.base import ORMModel, UTCModel
from app.schemas.response import BaseResponse


class ResourceBase(ORMModel):
    """Base resource payload carrying common attributes and metrics validation."""

    name: str = Field(..., min_length=1, max_length=100, description="Resource display name")
    provider_type: str = Field(..., min_length=1, max_length=50, description="Cloud provider or workload type")
    region: str = Field(..., min_length=1, max_length=50, description="Deployment region")
    provider_resource_id: str | None = Field(
        default=None,
        max_length=100,
        description="Provider-specific resource identifier (e.g. AWS instance ID)",
    )
    cpu_utilization: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage (0.0 - 100.0)")
    memory_utilization: float = Field(..., ge=0.0, le=100.0, description="Memory usage percentage (0.0 - 100.0)")
    storage_utilization: float = Field(default=0.0, ge=0.0, le=100.0, description="Storage usage percentage (0.0 - 100.0)")
    network_utilization: float = Field(default=0.0, ge=0.0, le=100.0, description="Network usage percentage (0.0 - 100.0)")
    current_capacity: float = Field(default=1.0, gt=0.0, description="Current capacity (e.g. vCPUs or instance count)")
    estimated_cost: float = Field(default=0.0, ge=0.0, description="Estimated monthly cost in USD")


class ResourceCreate(ResourceBase):
    """Payload for creating a new cloud resource."""

    pass


class ResourceUpdate(ORMModel):
    """Payload for updating an existing cloud resource.

    All fields are optional for partial updates.
    """

    name: str | None = Field(default=None, min_length=1, max_length=100)
    provider_type: str | None = Field(default=None, min_length=1, max_length=50)
    region: str | None = Field(default=None, min_length=1, max_length=50)
    cpu_utilization: float | None = Field(default=None, ge=0.0, le=100.0)
    memory_utilization: float | None = Field(default=None, ge=0.0, le=100.0)
    storage_utilization: float | None = Field(default=None, ge=0.0, le=100.0)
    network_utilization: float | None = Field(default=None, ge=0.0, le=100.0)
    current_capacity: float | None = Field(default=None, gt=0.0)
    estimated_cost: float | None = Field(default=None, ge=0.0)


class ResourceResponse(UTCModel):
    """Public representation of a cloud resource."""

    id: int
    user_id: int
    name: str
    provider_type: str
    region: str
    provider_resource_id: str | None
    cpu_utilization: float
    memory_utilization: float
    storage_utilization: float
    network_utilization: float
    current_capacity: float
    estimated_cost: float


class SyncResourceResponse(BaseResponse[dict]):
    """Response for provider resource synchronization."""

    success: bool = True
    message: str = "Resources synchronized successfully"
    data: dict = {
        "created": 0,
        "updated": 0,
        "unchanged": 0,
        "provider": "mock",
    }


class BorgReplayResponse(BaseResponse[dict]):
    """Response payload for Google Borg workload replay / ingestion."""

    success: bool = True
    message: str = "Borg workload replayed successfully"
    data: dict = {
        "dataset_source": "google_borg:borg_temporal_pairs.csv",
        "rows_selected": 0,
        "rows_imported": 0,
        "rows_skipped": 0,
        "duplicates_ignored": 0,
        "timestamp_range": {"start": None, "end": None},
        "resources_created": 0,
        "resources_updated": 0,
        "success": True,
        "message": "",
    }


class BorgDatasetSummaryResponse(BaseResponse[dict]):
    """Response payload for Google Borg dataset overview and live telemetry stats."""

    success: bool = True
    message: str = "Borg dataset summary retrieved successfully"
    data: dict = {
        "dataset_name": "Google Borg Workload Trace (Cluster Workloads)",
        "original_file": "data/temporal/borg_temporal_pairs.csv",
        "original_row_count": 65398,
        "original_column_count": 26,
        "imported_metric_count": 0,
        "imported_resource_count": 0,
        "mock_metric_count": 0,
        "mock_resource_count": 0,
        "earliest_timestamp": None,
        "latest_timestamp": None,
        "telemetry_source": "google_borg",
    }


__all__ = [
    "ResourceBase",
    "ResourceCreate",
    "ResourceUpdate",
    "ResourceResponse",
    "SyncResourceResponse",
    "BorgReplayResponse",
    "BorgDatasetSummaryResponse",
]
