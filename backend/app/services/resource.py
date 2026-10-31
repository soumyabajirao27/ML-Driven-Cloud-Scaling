"""ResourceService integration with Policy B Autoscaling Engine.

Wires PolicyBOptimizationEngine as the default engine for resource analysis,
providing full backwards compatibility with Stage 1 and Stage 2 while exposing
Policy B autoscaling recommendations, cooldown, hysteresis, and cost/power estimates.
"""

from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from app.core.exceptions import AuthorizationException, ResourceNotFoundException
from app.models.resource import CloudResource
from app.models.user import User
from app.repositories.metric import MetricRepository
from app.repositories.resource import ResourceRepository
from app.schemas.metric import MetricCreate, MetricResponse
from app.schemas.optimization import OptimizationAnalysis
from app.schemas.resource import ResourceCreate, ResourceUpdate
from app.services.metric import MetricService
from app.services.optimization import OptimizationService
from app.services.policy_b_engine import PolicyBOptimizationEngine

logger = logging.getLogger(__name__)


class ResourceService:
    """Business logic for cloud resource management and optimization."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = ResourceRepository(db)
        self.metric_repository = MetricRepository(db)
        # Use Policy B proactive optimization engine backed by RandomForestRegressor & baseline fallback
        self.optimization_service = OptimizationService(engine=PolicyBOptimizationEngine())

    def create_resource(self, user_id: int, resource_data: ResourceCreate) -> CloudResource:
        """Create a new cloud resource bound to the authenticated user ID."""
        resource = CloudResource(
            user_id=user_id,
            name=resource_data.name,
            provider_type=resource_data.provider_type,
            region=resource_data.region,
            cpu_utilization=resource_data.cpu_utilization,
            memory_utilization=resource_data.memory_utilization,
            storage_utilization=resource_data.storage_utilization,
            network_utilization=resource_data.network_utilization,
            current_capacity=resource_data.current_capacity,
            estimated_cost=resource_data.estimated_cost,
        )
        return self.repository.create(resource)

    def get_user_resources(self, current_user: User) -> list[CloudResource]:
        """Fetch all cloud resources visible to the user."""
        if current_user.is_superuser:
            return self.repository.get_all()
        return self.repository.get_user_resources(user_id=current_user.id)

    def get_resource_by_id(self, resource_id: int, current_user: User) -> CloudResource:
        """Fetch a resource by ID verifying ownership."""
        resource = self.repository.get_by_id(resource_id)
        if not resource:
            raise ResourceNotFoundException(f"CloudResource with ID {resource_id} not found")

        if not current_user.is_superuser and resource.user_id != current_user.id:
            raise AuthorizationException("Access denied to requested resource")

        return resource

    def update_resource(
        self,
        resource_id: int,
        resource_data: ResourceUpdate,
        current_user: User,
    ) -> CloudResource:
        """Update an existing cloud resource."""
        resource = self.get_resource_by_id(resource_id=resource_id, current_user=current_user)
        update_dict = resource_data.model_dump(exclude_unset=True)
        return self.repository.update(resource, update_dict)

    def delete_resource(self, resource_id: int, current_user: User) -> bool:
        """Delete an existing cloud resource."""
        resource = self.get_resource_by_id(resource_id=resource_id, current_user=current_user)
        return self.repository.delete(resource)

    def analyze_resource(self, resource_id: int, current_user: User) -> OptimizationAnalysis:
        """Run Phase 5 Policy B optimization analysis on a cloud resource."""
        resource = self.get_resource_by_id(resource_id=resource_id, current_user=current_user)
        metrics = self.metric_repository.get_metrics_for_resource(resource_id=resource_id)
        return self.optimization_service.analyze(resource, metrics)

    def create_metric(
        self,
        resource_id: int,
        metric_data: MetricCreate,
        current_user: User,
    ) -> MetricResponse:
        """Create a historical metric observation."""
        resource = self.get_resource_by_id(resource_id=resource_id, current_user=current_user)
        metric_service = MetricService(db=self.db)
        return metric_service.create_metric(resource=resource, metric_data=metric_data)

    def list_metrics(
        self,
        resource_id: int,
        current_user: User,
        limit: int | None = None,
        source: str | None = None,
    ):
        """List chronological metric samples."""
        self.get_resource_by_id(resource_id=resource_id, current_user=current_user)
        metric_service = MetricService(db=self.db)
        return metric_service.list_metrics(resource_id=resource_id, limit=limit, source=source)
