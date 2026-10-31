"""Google Borg Dataset Replay Service.

Coordinates safe, chronologically ordered ingestion of real Google Borg workload
traces into CloudScale AI's CloudResource and ResourceMetric tables.
Enforces safety limits (max 100 intervals per run), schema validation,
and idempotency to prevent duplicate metrics.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.exceptions import AppException, ValidationException
from app.models.metric import ResourceMetric
from app.models.resource import CloudResource
from app.models.user import User
from app.repositories.metric import MetricRepository
from app.repositories.resource import ResourceRepository

# Fixed reference base timestamp: Jan 1, 2026 UTC
BORG_EPOCH_BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
DEFAULT_TEMPORAL_PATH = Path("D:/PBL/data/temporal/borg_temporal_pairs.csv")
MAX_INTERVALS_PER_RUN = 100


@dataclass
class BorgReplayStats:
    """Summary of a Borg dataset ingestion / replay run."""

    dataset_source: str
    rows_selected: int
    rows_imported: int
    rows_skipped: int
    duplicates_ignored: int
    timestamp_range: dict[str, str | None]
    resources_created: int
    resources_updated: int
    success: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_source": self.dataset_source,
            "rows_selected": self.rows_selected,
            "rows_imported": self.rows_imported,
            "rows_skipped": self.rows_skipped,
            "duplicates_ignored": self.duplicates_ignored,
            "timestamp_range": self.timestamp_range,
            "resources_created": self.resources_created,
            "resources_updated": self.resources_updated,
            "success": self.success,
            "message": self.message,
        }


class BorgReplayService:
    """Safely ingests Google Borg temporal workload traces into PostgreSQL."""

    def __init__(self, db: Session, user: User, csv_path: Path | None = None) -> None:
        self.db = db
        self.user = user
        self.csv_path = csv_path or DEFAULT_TEMPORAL_PATH
        self.resource_repo = ResourceRepository(db)
        self.metric_repo = MetricRepository(db)

    def replay_workload(
        self,
        max_intervals: int = MAX_INTERVALS_PER_RUN,
        target_collection_id: str | None = None,
        target_instance_index: str | None = None,
    ) -> BorgReplayStats:
        """Read up to `max_intervals` consecutive chronological records and ingest them.

        Args:
            max_intervals: Maximum number of chronological intervals to process (capped at 100).
            target_collection_id: Optional filter for a specific collection ID.
            target_instance_index: Optional filter for a specific instance index.

        Returns:
            BorgReplayStats with execution metrics.
        """
        # Hard constraint on limit
        limit = min(max(1, max_intervals), MAX_INTERVALS_PER_RUN)

        if not self.csv_path.exists():
            raise AppException(
                message=f"Borg dataset file not found at {self.csv_path}",
                status_code=404,
            )

        selected_rows: list[dict[str, str]] = []
        skipped_count = 0

        # 1. Read and validate rows chronologically
        with open(self.csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Verify expected columns exist
            required_cols = {
                "timestamp_t_us",
                "time_hours",
                "collection_id",
                "instance_index",
                "cpu_utilization_pct_t",
                "memory_utilization_pct_t",
            }
            if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
                raise ValidationException(
                    message="Borg dataset CSV schema mismatch: missing required columns."
                )

            for row in reader:
                # Optional instance filter
                if target_collection_id and row.get("collection_id") != target_collection_id:
                    continue
                if target_instance_index and row.get("instance_index") != target_instance_index:
                    continue

                # Validate data integrity
                if not self._is_valid_row(row):
                    skipped_count += 1
                    continue

                selected_rows.append(row)
                if len(selected_rows) >= limit:
                    break

        if not selected_rows:
            return BorgReplayStats(
                dataset_source=str(self.csv_path),
                rows_selected=0,
                rows_imported=0,
                rows_skipped=skipped_count,
                duplicates_ignored=0,
                timestamp_range={"start": None, "end": None},
                resources_created=0,
                resources_updated=0,
                success=True,
                message="No valid rows found to ingest for the given filter.",
            )

        # 2. Ingest into PostgreSQL with resource mapping & deduplication
        imported_count = 0
        duplicate_count = 0
        resources_created = 0
        resources_updated = 0

        first_ts_str: str | None = None
        last_ts_str: str | None = None

        # Cache existing user resources by provider_resource_id to avoid repeated queries
        existing_resources = self.resource_repo.get_user_resources(self.user.id)
        resource_map: dict[str, CloudResource] = {
            r.provider_resource_id: r for r in existing_resources if r.provider_resource_id
        }

        # To prevent duplicate metrics in this run and against the DB, fetch recent metric timestamps
        # for affected resources
        for row in selected_rows:
            col_id = row["collection_id"]
            inst_idx = row["instance_index"]
            provider_res_id = f"borg-col-{col_id}-inst-{inst_idx}"
            cluster_id = row.get("cluster", "0")
            region_name = f"borg-cluster-{cluster_id}"
            res_name = f"borg-{col_id[:6]}-{inst_idx}"

            # Parse timestamps: timestamp_t_us is microseconds from trace start
            us_val = float(row["timestamp_t_us"])
            delta_sec = us_val / 1_000_000.0
            metric_time = BORG_EPOCH_BASE + timedelta(seconds=delta_sec)
            metric_time_iso = metric_time.isoformat()

            if first_ts_str is None:
                first_ts_str = metric_time_iso
            last_ts_str = metric_time_iso

            # Parse utilizations (safely bounded)
            cpu_val = max(0.0, float(row["cpu_utilization_pct_t"]))
            mem_val = max(0.0, float(row["memory_utilization_pct_t"]))

            # Get or create CloudResource
            resource = resource_map.get(provider_res_id)
            if not resource:
                # Derive capacity from cpu_request (normalized, default 1.0)
                cpu_req_raw = float(row.get("cpu_request") or 0.0)
                capacity = max(1.0, round(cpu_req_raw * 100.0, 2)) if cpu_req_raw > 0 else 1.0
                est_cost = round(capacity * 35.0, 2)  # Representative cost estimate

                resource = CloudResource(
                    user_id=self.user.id,
                    provider_resource_id=provider_res_id,
                    name=res_name,
                    provider_type="google_borg",
                    region=region_name,
                    cpu_utilization=cpu_val,
                    memory_utilization=mem_val,
                    storage_utilization=0.0,
                    network_utilization=0.0,
                    current_capacity=capacity,
                    estimated_cost=est_cost,
                )
                self.db.add(resource)
                self.db.flush()
                resource_map[provider_res_id] = resource
                resources_created += 1
            else:
                # Update latest telemetry snapshot
                resource.cpu_utilization = cpu_val
                resource.memory_utilization = mem_val
                resources_updated += 1

            # Check if metric with this timestamp already exists for this resource
            existing_metric = (
                self.db.query(ResourceMetric.id)
                .filter(
                    ResourceMetric.resource_id == resource.id,
                    ResourceMetric.timestamp == metric_time,
                )
                .first()
            )

            if existing_metric is not None:
                duplicate_count += 1
                continue

            # Insert ResourceMetric
            metric = ResourceMetric(
                resource_id=resource.id,
                timestamp=metric_time,
                cpu_utilization=cpu_val,
                memory_utilization=mem_val,
                storage_utilization=0.0,
                network_utilization=0.0,
                source="google_borg",
            )
            self.db.add(metric)
            imported_count += 1

        self.db.commit()

        return BorgReplayStats(
            dataset_source="google_borg:borg_temporal_pairs.csv",
            rows_selected=len(selected_rows),
            rows_imported=imported_count,
            rows_skipped=skipped_count,
            duplicates_ignored=duplicate_count,
            timestamp_range={"start": first_ts_str, "end": last_ts_str},
            resources_created=resources_created,
            resources_updated=resources_updated,
            success=True,
            message=(
                f"Successfully ingested {imported_count} Borg workload intervals "
                f"across {len(selected_rows)} selected rows ({duplicate_count} duplicates ignored)."
            ),
        )

    def _is_valid_row(self, row: dict[str, str]) -> bool:
        """Validate that a CSV row has non-empty, finite numeric values for key columns."""
        try:
            t_us = float(row.get("timestamp_t_us", ""))
            cpu = float(row.get("cpu_utilization_pct_t", ""))
            mem = float(row.get("memory_utilization_pct_t", ""))

            if math.isnan(t_us) or math.isnan(cpu) or math.isnan(mem):
                return False
            if math.isinf(t_us) or math.isinf(cpu) or math.isinf(mem):
                return False
            if t_us < 0 or cpu < 0 or mem < 0:
                return False

            if not row.get("collection_id") or not row.get("instance_index"):
                return False

            return True
        except (ValueError, TypeError):
            return False
