"""SQLAlchemy ORM models for Energy Orchestrator."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Sample(Base):
    """Raw sensor data samples."""

    __tablename__ = "samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(128), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(32), nullable=True)


class SensorMapping(Base):
    """Mapping between logical categories and Home Assistant entities."""

    __tablename__ = "sensor_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(64), nullable=False, index=True)
    entity_id = Column(String(128), nullable=False, index=True)
    is_active = Column(Boolean, nullable=False, default=True)
    priority = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        UniqueConstraint("category", "entity_id", name="uq_category_entity"),
    )


class ResampledSample(Base):
    """Resampled values per 5-minute slot per category."""

    __tablename__ = "resampled_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    slot_start = Column(DateTime, nullable=False, index=True)
    category = Column(String(64), nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(32), nullable=True)

    __table_args__ = (
        UniqueConstraint("slot_start", "category", name="uq_slot_category"),
    )
