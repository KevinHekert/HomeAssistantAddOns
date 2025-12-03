from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Double, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, Mapped, mapped_column


Base = declarative_base()


def _utcnow():
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        UniqueConstraint("entity_id", "timestamp", name="uq_entity_timestamp"),
    )


class SyncStatus(Base):
    __tablename__ = "sync_status"

    entity_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    last_attempt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_success: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SensorMapping(Base):
    """Maps logical categories to Home Assistant entity IDs."""
    __tablename__ = "sensor_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    __table_args__ = (
        UniqueConstraint("category", "entity_id", name="uq_category_entity"),
    )


class ResampledSample(Base):
    """Stores resampled values per 5-minute slot per category.
    
    The is_derived field indicates whether this sample is:
    - False: Direct resampling from raw sensor data
    - True: Calculated/derived from other resampled data (e.g., virtual sensors, averages)
    """
    __tablename__ = "resampled_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slot_start: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Double, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_derived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")

    __table_args__ = (
        UniqueConstraint("slot_start", "category", name="uq_slot_category"),
    )


class FeatureStatistic(Base):
    """Stores time-span average statistics calculated from resampled data.
    
    This table stores rolling time-window averages (avg_1h, avg_6h, avg_24h, avg_7d)
    calculated from the resampled_samples table. These statistics are:
    - Calculated AFTER raw sensor resampling and virtual sensor calculations
    - Used as features for model training when enabled in configuration
    - Separate from raw/virtual sensor data for clarity and performance
    
    Examples:
    - outdoor_temp_avg_1h: 1-hour rolling average of outdoor temperature
    - indoor_temp_avg_24h: 24-hour rolling average of indoor temperature
    - temp_delta_avg_6h: 6-hour rolling average of the virtual temp_delta sensor
    """
    __tablename__ = "feature_statistics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slot_start: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    sensor_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    stat_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # avg_1h, avg_6h, avg_24h, avg_7d
    value: Mapped[float] = mapped_column(Double, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # Number of samples used

    __table_args__ = (
        UniqueConstraint("slot_start", "sensor_name", "stat_type", name="uq_feature_stat"),
    )


class OptimizerRun(Base):
    """Stores optimization run metadata and overall status."""
    __tablename__ = "optimizer_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    phase: Mapped[str] = mapped_column(String(32), nullable=False)  # "initializing", "training", "complete", "error"
    total_configurations: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_configurations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_result_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class OptimizerResult(Base):
    """Stores individual optimizer configuration test results."""
    __tablename__ = "optimizer_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    config_name: Mapped[str] = mapped_column(String(256), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)  # "single_step" or "two_step"
    experimental_features_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON dict
    val_mape_pct: Mapped[float | None] = mapped_column(Double, nullable=True)
    val_mae_kwh: Mapped[float | None] = mapped_column(Double, nullable=True)
    val_r2: Mapped[float | None] = mapped_column(Double, nullable=True)
    train_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    val_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class OptimizerConfig(Base):
    """Stores optimizer configuration settings."""
    __tablename__ = "optimizer_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    max_workers: Mapped[int | None] = mapped_column(Integer, nullable=True)  # None or 0 = auto-calculate
    max_combinations: Mapped[int | None] = mapped_column(Integer, nullable=True)  # None = default (1024), limits feature combination explosion
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)