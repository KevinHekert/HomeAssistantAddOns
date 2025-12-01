from datetime import datetime

from sqlalchemy import Boolean, DateTime, Double, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import declarative_base, Mapped, mapped_column


Base = declarative_base()


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
    """Stores resampled values per 5-minute slot per category."""
    __tablename__ = "resampled_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slot_start: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Double, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        UniqueConstraint("slot_start", "category", name="uq_slot_category"),
    )