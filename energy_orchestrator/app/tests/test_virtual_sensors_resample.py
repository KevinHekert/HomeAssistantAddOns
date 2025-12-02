"""
Tests for virtual sensor resampling functionality.

Verifies that virtual sensors are calculated correctly during the resampling process.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, ResampledSample, Sample, SensorMapping
from db.resample import resample_all_categories
from db.virtual_sensors import (
    VirtualSensorDefinition,
    VirtualSensorOperation,
    VirtualSensorsConfiguration,
    reset_virtual_sensors_config,
)
import db.core as core_module
import db.resample as resample_module
import db.virtual_sensors as virtual_sensors_module


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for configuration files."""
    return tmp_path


@pytest.fixture
def patch_engine_and_config(test_engine, temp_config_dir, monkeypatch):
    """Patch the engine and configuration directory."""
    # Patch engine
    monkeypatch.setattr(core_module, "engine", test_engine)
    monkeypatch.setattr(resample_module, "engine", test_engine)

    # Patch init_db_schema
    def mock_init_db_schema():
        Base.metadata.create_all(test_engine)

    monkeypatch.setattr(core_module, "init_db_schema", mock_init_db_schema)
    monkeypatch.setattr(resample_module, "init_db_schema", mock_init_db_schema)

    # Patch DATA_DIR and config file path
    monkeypatch.setattr(virtual_sensors_module, "DATA_DIR", temp_config_dir)
    monkeypatch.setattr(
        virtual_sensors_module,
        "VIRTUAL_SENSORS_CONFIG_FILE",
        temp_config_dir / "virtual_sensors_config.json",
    )
    
    # Reset global config using the public reset function
    reset_virtual_sensors_config()

    return test_engine


class TestVirtualSensorResampling:
    """Test that virtual sensors are calculated during resampling."""

    def test_subtract_operation(self, patch_engine_and_config, temp_config_dir):
        """Test virtual sensor with subtract operation (e.g., temp_delta = target - indoor)."""
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="target_temp",
                    entity_id="sensor.target_temp",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="indoor_temp",
                    entity_id="sensor.indoor_temp",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data - need at least 2 samples to establish a range
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            # Target temp: 21°C
            session.add(
                Sample(
                    entity_id="sensor.target_temp",
                    timestamp=base_time,
                    value=21.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.target_temp",
                    timestamp=base_time + timedelta(minutes=10),
                    value=21.0,
                    unit="°C",
                )
            )
            # Indoor temp: 19°C
            session.add(
                Sample(
                    entity_id="sensor.indoor_temp",
                    timestamp=base_time,
                    value=19.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.indoor_temp",
                    timestamp=base_time + timedelta(minutes=10),
                    value=19.0,
                    unit="°C",
                )
            )
            session.commit()

        # Configure virtual sensor
        config = VirtualSensorsConfiguration()
        virtual_sensor = VirtualSensorDefinition(
            name="temp_delta",
            display_name="Temperature Delta",
            description="Difference between target and indoor temperature",
            source_sensor1="target_temp",
            source_sensor2="indoor_temp",
            operation=VirtualSensorOperation.SUBTRACT,
            unit="°C",
            enabled=True,
        )
        config.add_sensor(virtual_sensor)
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify raw sensors were resampled
        with Session(patch_engine_and_config) as session:
            target_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "target_temp")
                .first()
            )
            assert target_sample is not None
            assert target_sample.value == 21.0

            indoor_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "indoor_temp")
                .first()
            )
            assert indoor_sample is not None
            assert indoor_sample.value == 19.0

            # Verify virtual sensor was calculated
            virtual_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "temp_delta")
                .first()
            )
            assert virtual_sample is not None
            assert virtual_sample.value == 2.0  # 21 - 19 = 2
            assert virtual_sample.unit == "°C"

    def test_add_operation(self, patch_engine_and_config, temp_config_dir):
        """Test virtual sensor with add operation."""
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="power1",
                    entity_id="sensor.power1",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="power2",
                    entity_id="sensor.power2",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data - need at least 2 samples to establish a range
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            session.add(
                Sample(
                    entity_id="sensor.power1",
                    timestamp=base_time,
                    value=100.0,
                    unit="W",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.power1",
                    timestamp=base_time + timedelta(minutes=10),
                    value=100.0,
                    unit="W",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.power2",
                    timestamp=base_time,
                    value=150.0,
                    unit="W",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.power2",
                    timestamp=base_time + timedelta(minutes=10),
                    value=150.0,
                    unit="W",
                )
            )
            session.commit()

        # Configure virtual sensor
        config = VirtualSensorsConfiguration()
        virtual_sensor = VirtualSensorDefinition(
            name="total_power",
            display_name="Total Power",
            description="Sum of two power sensors",
            source_sensor1="power1",
            source_sensor2="power2",
            operation=VirtualSensorOperation.ADD,
            unit="W",
            enabled=True,
        )
        config.add_sensor(virtual_sensor)
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify virtual sensor was calculated
        with Session(patch_engine_and_config) as session:
            virtual_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "total_power")
                .first()
            )
            assert virtual_sample is not None
            assert virtual_sample.value == 250.0  # 100 + 150 = 250
            assert virtual_sample.unit == "W"

    def test_multiply_operation(self, patch_engine_and_config, temp_config_dir):
        """Test virtual sensor with multiply operation."""
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="voltage",
                    entity_id="sensor.voltage",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="current",
                    entity_id="sensor.current",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data - need at least 2 samples to establish a range
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            session.add(
                Sample(
                    entity_id="sensor.voltage",
                    timestamp=base_time,
                    value=230.0,
                    unit="V",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.voltage",
                    timestamp=base_time + timedelta(minutes=10),
                    value=230.0,
                    unit="V",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.current",
                    timestamp=base_time,
                    value=5.0,
                    unit="A",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.current",
                    timestamp=base_time + timedelta(minutes=10),
                    value=5.0,
                    unit="A",
                )
            )
            session.commit()

        # Configure virtual sensor
        config = VirtualSensorsConfiguration()
        virtual_sensor = VirtualSensorDefinition(
            name="power",
            display_name="Power",
            description="Voltage times current",
            source_sensor1="voltage",
            source_sensor2="current",
            operation=VirtualSensorOperation.MULTIPLY,
            unit="W",
            enabled=True,
        )
        config.add_sensor(virtual_sensor)
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify virtual sensor was calculated
        with Session(patch_engine_and_config) as session:
            virtual_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "power")
                .first()
            )
            assert virtual_sample is not None
            assert virtual_sample.value == 1150.0  # 230 * 5 = 1150
            assert virtual_sample.unit == "W"

    def test_divide_operation(self, patch_engine_and_config, temp_config_dir):
        """Test virtual sensor with divide operation."""
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="output_energy",
                    entity_id="sensor.output_energy",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="input_energy",
                    entity_id="sensor.input_energy",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data - need at least 2 samples to establish a range
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            session.add(
                Sample(
                    entity_id="sensor.output_energy",
                    timestamp=base_time,
                    value=300.0,
                    unit="kWh",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.output_energy",
                    timestamp=base_time + timedelta(minutes=10),
                    value=300.0,
                    unit="kWh",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.input_energy",
                    timestamp=base_time,
                    value=100.0,
                    unit="kWh",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.input_energy",
                    timestamp=base_time + timedelta(minutes=10),
                    value=100.0,
                    unit="kWh",
                )
            )
            session.commit()

        # Configure virtual sensor
        config = VirtualSensorsConfiguration()
        virtual_sensor = VirtualSensorDefinition(
            name="efficiency",
            display_name="Efficiency",
            description="Output divided by input",
            source_sensor1="output_energy",
            source_sensor2="input_energy",
            operation=VirtualSensorOperation.DIVIDE,
            unit="",
            enabled=True,
        )
        config.add_sensor(virtual_sensor)
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify virtual sensor was calculated
        with Session(patch_engine_and_config) as session:
            virtual_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "efficiency")
                .first()
            )
            assert virtual_sample is not None
            assert virtual_sample.value == 3.0  # 300 / 100 = 3
            assert virtual_sample.unit == ""

    def test_average_operation(self, patch_engine_and_config, temp_config_dir):
        """Test virtual sensor with average operation."""
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="temp1",
                    entity_id="sensor.temp1",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="temp2",
                    entity_id="sensor.temp2",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data - need at least 2 samples to establish a range
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            session.add(
                Sample(
                    entity_id="sensor.temp1",
                    timestamp=base_time,
                    value=20.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.temp1",
                    timestamp=base_time + timedelta(minutes=10),
                    value=20.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.temp2",
                    timestamp=base_time,
                    value=22.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.temp2",
                    timestamp=base_time + timedelta(minutes=10),
                    value=22.0,
                    unit="°C",
                )
            )
            session.commit()

        # Configure virtual sensor
        config = VirtualSensorsConfiguration()
        virtual_sensor = VirtualSensorDefinition(
            name="avg_temp",
            display_name="Average Temperature",
            description="Average of two temperatures",
            source_sensor1="temp1",
            source_sensor2="temp2",
            operation=VirtualSensorOperation.AVERAGE,
            unit="°C",
            enabled=True,
        )
        config.add_sensor(virtual_sensor)
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify virtual sensor was calculated
        with Session(patch_engine_and_config) as session:
            virtual_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "avg_temp")
                .first()
            )
            assert virtual_sample is not None
            assert virtual_sample.value == 21.0  # (20 + 22) / 2 = 21
            assert virtual_sample.unit == "°C"

    def test_missing_source_sensor_skips_virtual(self, patch_engine_and_config, temp_config_dir):
        """Test that virtual sensor is skipped when source sensor is missing."""
        # Create sensor mappings - only one sensor
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="target_temp",
                    entity_id="sensor.target_temp",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data for only one sensor - need at least 2 samples to establish a range
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            session.add(
                Sample(
                    entity_id="sensor.target_temp",
                    timestamp=base_time,
                    value=21.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.target_temp",
                    timestamp=base_time + timedelta(minutes=10),
                    value=21.0,
                    unit="°C",
                )
            )
            session.commit()

        # Configure virtual sensor that needs both sensors
        config = VirtualSensorsConfiguration()
        virtual_sensor = VirtualSensorDefinition(
            name="temp_delta",
            display_name="Temperature Delta",
            description="Difference between target and indoor temperature",
            source_sensor1="target_temp",
            source_sensor2="indoor_temp",  # This one is missing
            operation=VirtualSensorOperation.SUBTRACT,
            unit="°C",
            enabled=True,
        )
        config.add_sensor(virtual_sensor)
        config.save()

        # Resample - should complete but skip virtual sensor
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify that no virtual sensor was calculated
        with Session(patch_engine_and_config) as session:
            virtual_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "temp_delta")
                .first()
            )
            assert virtual_sample is None  # Should not exist

    def test_disabled_virtual_sensor_not_calculated(self, patch_engine_and_config, temp_config_dir):
        """Test that disabled virtual sensors are not calculated."""
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="target_temp",
                    entity_id="sensor.target_temp",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="indoor_temp",
                    entity_id="sensor.indoor_temp",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data - need at least 2 samples to establish a range
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            session.add(
                Sample(
                    entity_id="sensor.target_temp",
                    timestamp=base_time,
                    value=21.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.target_temp",
                    timestamp=base_time + timedelta(minutes=10),
                    value=21.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.indoor_temp",
                    timestamp=base_time,
                    value=19.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.indoor_temp",
                    timestamp=base_time + timedelta(minutes=10),
                    value=19.0,
                    unit="°C",
                )
            )
            session.commit()

        # Configure virtual sensor as DISABLED
        config = VirtualSensorsConfiguration()
        virtual_sensor = VirtualSensorDefinition(
            name="temp_delta",
            display_name="Temperature Delta",
            description="Difference between target and indoor temperature",
            source_sensor1="target_temp",
            source_sensor2="indoor_temp",
            operation=VirtualSensorOperation.SUBTRACT,
            unit="°C",
            enabled=False,  # Disabled
        )
        config.add_sensor(virtual_sensor)
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify that no virtual sensor was calculated
        with Session(patch_engine_and_config) as session:
            virtual_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "temp_delta")
                .first()
            )
            assert virtual_sample is None  # Should not exist because disabled

    def test_multiple_virtual_sensors(self, patch_engine_and_config, temp_config_dir):
        """Test multiple virtual sensors calculated in same resampling run."""
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="target_temp",
                    entity_id="sensor.target_temp",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="indoor_temp",
                    entity_id="sensor.indoor_temp",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="outdoor_temp",
                    entity_id="sensor.outdoor_temp",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data - need at least 2 samples to establish a range
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            session.add(
                Sample(
                    entity_id="sensor.target_temp",
                    timestamp=base_time,
                    value=21.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.target_temp",
                    timestamp=base_time + timedelta(minutes=10),
                    value=21.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.indoor_temp",
                    timestamp=base_time,
                    value=19.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.indoor_temp",
                    timestamp=base_time + timedelta(minutes=10),
                    value=19.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.outdoor_temp",
                    timestamp=base_time,
                    value=5.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.outdoor_temp",
                    timestamp=base_time + timedelta(minutes=10),
                    value=5.0,
                    unit="°C",
                )
            )
            session.commit()

        # Configure multiple virtual sensors
        config = VirtualSensorsConfiguration()
        config.add_sensor(
            VirtualSensorDefinition(
                name="temp_delta",
                display_name="Temperature Delta",
                description="Target minus indoor",
                source_sensor1="target_temp",
                source_sensor2="indoor_temp",
                operation=VirtualSensorOperation.SUBTRACT,
                unit="°C",
                enabled=True,
            )
        )
        config.add_sensor(
            VirtualSensorDefinition(
                name="indoor_outdoor_delta",
                display_name="Indoor-Outdoor Delta",
                description="Indoor minus outdoor",
                source_sensor1="indoor_temp",
                source_sensor2="outdoor_temp",
                operation=VirtualSensorOperation.SUBTRACT,
                unit="°C",
                enabled=True,
            )
        )
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify both virtual sensors were calculated
        with Session(patch_engine_and_config) as session:
            temp_delta_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "temp_delta")
                .first()
            )
            assert temp_delta_sample is not None
            assert temp_delta_sample.value == 2.0  # 21 - 19 = 2

            indoor_outdoor_delta_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "indoor_outdoor_delta")
                .first()
            )
            assert indoor_outdoor_delta_sample is not None
            assert indoor_outdoor_delta_sample.value == 14.0  # 19 - 5 = 14

    def test_division_by_zero_returns_none(self, patch_engine_and_config, temp_config_dir):
        """Test that division by zero is handled gracefully."""
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="numerator",
                    entity_id="sensor.numerator",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="denominator",
                    entity_id="sensor.denominator",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data with zero denominator - need at least 2 samples to establish a range
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            session.add(
                Sample(
                    entity_id="sensor.numerator",
                    timestamp=base_time,
                    value=100.0,
                    unit="",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.numerator",
                    timestamp=base_time + timedelta(minutes=10),
                    value=100.0,
                    unit="",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.denominator",
                    timestamp=base_time,
                    value=0.0,  # Zero!
                    unit="",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.denominator",
                    timestamp=base_time + timedelta(minutes=10),
                    value=0.0,  # Zero!
                    unit="",
                )
            )
            session.commit()

        # Configure virtual sensor with divide operation
        config = VirtualSensorsConfiguration()
        virtual_sensor = VirtualSensorDefinition(
            name="ratio",
            display_name="Ratio",
            description="Division test",
            source_sensor1="numerator",
            source_sensor2="denominator",
            operation=VirtualSensorOperation.DIVIDE,
            unit="",
            enabled=True,
        )
        config.add_sensor(virtual_sensor)
        config.save()

        # Resample - should not crash
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify that no virtual sensor value was stored (because division by zero)
        with Session(patch_engine_and_config) as session:
            virtual_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "ratio")
                .first()
            )
            assert virtual_sample is None  # Should not exist due to division by zero
