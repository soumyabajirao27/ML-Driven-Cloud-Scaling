"""Resource metric repository implementation backed by SQLAlchemy 2.0."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.metric import ResourceMetric
from app.repositories.base import BaseRepository


class MetricRepository(BaseRepository[ResourceMetric]):
    """Repository for the ``resource_metrics`` table."""

    def __init__(self, db: Session) -> None:
        super().__init__(db=db, model=ResourceMetric)

    def get_metrics_for_resource(
        self,
        resource_id: int,
        limit: int | None = None,
        source: str | None = None,
    ) -> list[ResourceMetric]:
        """Fetch metrics for a resource ordered by timestamp ascending."""
        statement = (
            select(ResourceMetric)
            .where(ResourceMetric.resource_id == resource_id)
        )
        if source:
            statement = statement.where(ResourceMetric.source == source)
        statement = statement.order_by(ResourceMetric.timestamp.asc())
        if limit is not None:
            statement = statement.limit(limit)
        return list(self.db.execute(statement).scalars().all())

    def count_for_resource(self, resource_id: int) -> int:
        """Return the number of metric samples for a given resource."""
        from sqlalchemy import func
        statement = (
            select(func.count())
            .select_from(ResourceMetric)
            .where(ResourceMetric.resource_id == resource_id)
        )
        return int(self.db.execute(statement).scalar_one())
