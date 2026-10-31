"""Application configuration loaded via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings sourced from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "CloudScale AI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    # Database
    DATABASE_URL: str
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_PRE_PING: bool = True
    DB_POOL_RECYCLE: int = 3600

    # JWT Authentication
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production-1234567890"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Cloud Provider Integration
    CLOUD_PROVIDER: str = "mock"

    # ML Inference Settings
    ML_MODEL_TYPE: str = "random_forest"
    ML_MODEL_PATH: str | None = None
    ML_CONFIDENCE_THRESHOLD: float = 0.80

    # Phase 5 Policy B & Autoscaling Settings
    POLICY_B_COOLDOWN_STEPS: int = 2
    POLICY_B_COOLDOWN_SECONDS: int = 300
    POLICY_B_HYSTERESIS_BUFFER_PCT: float = 2.0
    POLICY_B_UNDERUTILIZED_PCT: float = 20.0
    POLICY_B_OVERLOADED_PCT: float = 85.0
    POLICY_B_MIN_CPU_ALLOCATION: float = 0.001
    POLICY_B_MAX_CPU_ALLOCATION: float = 0.200

    # Power and Cost Simulation Parameters
    SIM_CPU_PRICE_PER_UNIT: float = 1.0
    SIM_MEM_PRICE_PER_UNIT: float = 0.5
    SIM_IDLE_POWER_W: float = 100.0
    SIM_MAX_POWER_W: float = 250.0
    SIM_SCALE_UP_LATENCY_SEC: float = 90.0
    SIM_SCALE_DOWN_LATENCY_SEC: float = 30.0


settings = Settings()
