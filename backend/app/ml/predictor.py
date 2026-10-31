"""ML prediction service for CloudScale AI demand prediction.

Loads a trained ``RandomForestRegressor`` artifact, validates input features,
verifies temporal ordering and data sufficiency from historical metrics,
and returns structured predictions with complete provenance metadata.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np

from app.core.config import settings
from app.ml.features import prepare_single_prediction_features
from app.ml.trainer import MODEL_DIR, MODEL_FILENAME
from app.models.metric import ResourceMetric


class ModelNotAvailableError(Exception):
    """Raised when a trained model artifact is missing or cannot be loaded."""


class PredictionService:
    """Service that loads a trained ML model and produces numeric predictions."""

    EXPECTED_FEATURES_COUNT = 9
    MODEL_ID = "random_forest_v1"
    MODEL_TYPE = "RandomForestRegressor"
    TARGET = "next_sample_cpu_utilization_pct"
    MODEL_VERSION = "1.0.0"

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or settings.ML_MODEL_PATH or os.path.join(MODEL_DIR, MODEL_FILENAME)
        self._model = None

    def _load_model(self):
        """Lazily load and validate the trained model artifact from disk."""
        if self._model is not None:
            return self._model
        if not os.path.exists(self.model_path):
            raise ModelNotAvailableError(f"Trained model artifact not found at {self.model_path}")
        try:
            loaded = joblib.load(self.model_path)
            # Verify feature dimension compatibility
            n_features = getattr(loaded, "n_features_in_", None)
            if n_features is not None and n_features != self.EXPECTED_FEATURES_COUNT:
                raise ModelNotAvailableError(
                    f"Model expects {n_features} features, but pipeline provides {self.EXPECTED_FEATURES_COUNT}"
                )
            self._model = loaded
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelNotAvailableError(f"Failed to load or validate model artifact: {exc}")
        return self._model

    def is_available(self) -> bool:
        """Return True if the trained model artifact exists and can be loaded."""
        try:
            self._load_model()
            return True
        except ModelNotAvailableError:
            return False

    def predict_cpu_utilization(self, metrics: list[ResourceMetric]) -> float:
        """Return a numeric prediction of future CPU utilization percentage.

        Args:
            metrics: Chronologically ordered metric samples (most recent last).

        Returns:
            Predicted CPU utilization percentage (float clamped between 0.0 and 100.0).

        Raises:
            ModelNotAvailableError: If the model artifact is missing or invalid.
            ValueError: If fewer than required metrics are provided.
        """
        if not metrics:
            raise ValueError("No metrics provided for prediction")

        model = self._load_model()
        features = prepare_single_prediction_features(metrics)
        prediction = model.predict(features)
        return float(np.clip(prediction[0], 0.0, 100.0))

    def get_provenance(self, metrics: list[ResourceMetric] | None) -> dict[str, Any]:
        """Return provenance metadata describing the model and evaluation context."""
        sample_count = len(metrics) if metrics else 0
        data_sources = set(m.source for m in metrics) if metrics else set()
        source_str = ",".join(sorted(data_sources)) if data_sources else "unknown"

        return {
            "model_id": self.MODEL_ID,
            "model_type": self.MODEL_TYPE,
            "prediction_target": self.TARGET,
            "input_data_source": source_str,
            "is_fallback": False,
            "sufficient_history": sample_count >= 3,
            "sample_count_used": sample_count,
            "model_version": self.MODEL_VERSION,
        }
