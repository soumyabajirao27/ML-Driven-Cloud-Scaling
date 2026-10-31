"""Thread-safe state manager for autoscaling cooldown and hysteresis.

Manages persistent-in-process state for CloudScale AI Policy B scaling decisions,
tracking cooldown steps, action timestamps, and recent scaling directions
to prevent thrashing or rapid oscillations.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ResourceScalingState:
    """State tracking for a single managed resource."""

    resource_id: int
    cooldown_steps_remaining: int = 0
    last_action: str = "MAINTAIN"
    last_action_timestamp: float = 0.0
    last_action_datetime: datetime | None = None
    last_allocated_capacity: float = 1.0
    action_count: int = 0


class ScalingStateManager:
    """Thread-safe coordinator for resource autoscaling cooldown and history.

    Features:
    - Thread-safe access via re-entrant lock (RLock).
    - Tracks both discrete steps (as in Phase 5 simulation) and wall-clock seconds.
    - Prevents concurrent requests from bypassing cooldown checks.
    - Supports state introspection and programmatic reset for testing.
    """

    _instance: ScalingStateManager | None = None
    _lock = threading.RLock()

    def __new__(cls) -> ScalingStateManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._states: dict[int, ResourceScalingState] = {}
                cls._instance._cooldown_steps = settings.POLICY_B_COOLDOWN_STEPS
                cls._instance._cooldown_seconds = settings.POLICY_B_COOLDOWN_SECONDS
            return cls._instance

    def get_state(self, resource_id: int) -> ResourceScalingState:
        """Retrieve or initialize the state for a specific resource."""
        with self._lock:
            if resource_id not in self._states:
                self._states[resource_id] = ResourceScalingState(resource_id=resource_id)
            return self._states[resource_id]

    def check_cooldown(self, resource_id: int) -> tuple[bool, int, float]:
        """Check if a resource is currently in cooldown.

        Returns:
            Tuple of (is_in_cooldown, remaining_steps, remaining_seconds).
        """
        with self._lock:
            state = self.get_state(resource_id)
            now = time.time()
            elapsed_sec = now - state.last_action_timestamp if state.last_action_timestamp > 0 else 999999.0

            # Step-based cooldown check
            step_active = state.cooldown_steps_remaining > 0
            
            # Time-based cooldown check
            time_remaining_sec = max(0.0, self._cooldown_seconds - elapsed_sec) if state.last_action_timestamp > 0 else 0.0
            time_active = time_remaining_sec > 0.0 and state.last_action in ("SCALE_UP", "SCALE_DOWN")

            is_active = step_active or (time_active and state.cooldown_steps_remaining > 0)

            return is_active, state.cooldown_steps_remaining, round(time_remaining_sec, 1)

    def record_action(
        self,
        resource_id: int,
        action: str,
        allocated_capacity: float,
        is_simulated: bool = True,
    ) -> ResourceScalingState:
        """Record an executed or simulated scaling action and update cooldown counters.

        If action is SCALE_UP or SCALE_DOWN, resets cooldown counter to configured steps.
        If action is MAINTAIN or DEFERRED, decrements cooldown counter if greater than zero.
        """
        with self._lock:
            state = self.get_state(resource_id)
            now = time.time()

            if action in ("SCALE_UP", "SCALE_DOWN"):
                state.cooldown_steps_remaining = self._cooldown_steps
                state.last_action = action
                state.last_action_timestamp = now
                state.last_action_datetime = datetime.utcnow()
                state.last_allocated_capacity = allocated_capacity
                state.action_count += 1
                logger.info(
                    "Autoscaling action '%s' recorded for resource %d. Cooldown set to %d steps.",
                    action,
                    resource_id,
                    self._cooldown_steps,
                )
            else:
                # MAINTAIN or deferred: progress the cooldown
                if state.cooldown_steps_remaining > 0:
                    state.cooldown_steps_remaining -= 1

            return state

    def reset(self, resource_id: int | None = None) -> None:
        """Reset state for a specific resource or all resources."""
        with self._lock:
            if resource_id is not None:
                if resource_id in self._states:
                    del self._states[resource_id]
            else:
                self._states.clear()


scaling_state_manager = ScalingStateManager()

