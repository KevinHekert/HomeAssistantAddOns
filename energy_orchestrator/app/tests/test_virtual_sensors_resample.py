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


class TestIsDerivedFlag:
    """Test that is_derived flag is set correctly during resampling."""

    def test_raw_sensors_marked_as_not_derived(self, patch_engine_and_config, temp_config_dir):
        """Test that raw sensor samples have is_derived=False."""
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="outdoor_temp",
                    entity_id="sensor.outdoor_temp",
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

        # Add sample data
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
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
            session.add(
                Sample(
                    entity_id="sensor.indoor_temp",
                    timestamp=base_time,
                    value=20.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.indoor_temp",
                    timestamp=base_time + timedelta(minutes=10),
                    value=20.0,
                    unit="°C",
                )
            )
            session.commit()

        # Resample without any virtual sensors
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify all resampled samples are marked as not derived
        with Session(patch_engine_and_config) as session:
            all_samples = session.query(ResampledSample).all()
            assert len(all_samples) > 0, "Should have resampled samples"
            
            for sample in all_samples:
                assert sample.is_derived is False, f"Raw sensor {sample.category} should have is_derived=False"

    def test_virtual_sensors_marked_as_derived(self, patch_engine_and_config, temp_config_dir):
        """Test that virtual sensor samples have is_derived=True."""
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

        # Add sample data
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

        # Configure a virtual sensor
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
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify is_derived flags
        with Session(patch_engine_and_config) as session:
            # Check raw sensors
            target_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "target_temp")
                .first()
            )
            assert target_sample is not None
            assert target_sample.is_derived is False, "Raw sensor should have is_derived=False"
            
            indoor_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "indoor_temp")
                .first()
            )
            assert indoor_sample is not None
            assert indoor_sample.is_derived is False, "Raw sensor should have is_derived=False"
            
            # Check virtual sensor
            virtual_sample = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "temp_delta")
                .first()
            )
            assert virtual_sample is not None
            assert virtual_sample.is_derived is True, "Virtual sensor should have is_derived=True"
            assert virtual_sample.value == 2.0  # 21 - 19 = 2

    def test_mixed_sensors_have_correct_flags(self, patch_engine_and_config, temp_config_dir):
        """Test that a mix of raw and virtual sensors have correct is_derived flags."""
        # Create sensor mappings for 3 raw sensors
        with Session(patch_engine_and_config) as session:
            for category, entity_id in [
                ("outdoor_temp", "sensor.outdoor_temp"),
                ("indoor_temp", "sensor.indoor_temp"),
                ("target_temp", "sensor.target_temp"),
            ]:
                session.add(
                    SensorMapping(
                        category=category,
                        entity_id=entity_id,
                        is_active=True,
                        priority=1,
                    )
                )
            session.commit()

        # Add sample data for a single time slot
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            for entity_id, value in [
                ("sensor.outdoor_temp", 5.0),
                ("sensor.indoor_temp", 20.0),
                ("sensor.target_temp", 21.0),
            ]:
                # Add samples at the start and end of the 10-minute window to create 1 slot
                session.add(
                    Sample(
                        entity_id=entity_id,
                        timestamp=base_time,
                        value=value,
                        unit="°C",
                    )
                )
                session.add(
                    Sample(
                        entity_id=entity_id,
                        timestamp=base_time + timedelta(minutes=5),
                        value=value,
                        unit="°C",
                    )
                )
            session.commit()

        # Configure 2 virtual sensors
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

        # Verify counts and flags
        with Session(patch_engine_and_config) as session:
            all_samples = session.query(ResampledSample).all()
            # We create 1 time slot, so: 3 raw sensors + 2 virtual sensors = 5 samples total
            assert len(all_samples) == 5, f"Should have 3 raw + 2 virtual = 5 samples, got {len(all_samples)}"
            
            raw_samples = [s for s in all_samples if not s.is_derived]
            derived_samples = [s for s in all_samples if s.is_derived]
            
            assert len(raw_samples) == 3, "Should have 3 raw sensor samples"
            assert len(derived_samples) == 2, "Should have 2 virtual sensor samples"
            
            # Verify raw sensor categories
            raw_categories = {s.category for s in raw_samples}
            assert raw_categories == {"outdoor_temp", "indoor_temp", "target_temp"}
            
            # Verify virtual sensor categories
            derived_categories = {s.category for s in derived_samples}
            assert derived_categories == {"temp_delta", "indoor_outdoor_delta"}

    def test_virtual_sensors_calculated_for_all_available_timeframes(
        self, patch_engine_and_config, temp_config_dir
    ):
        """Test that virtual sensors are calculated for every timeframe where source data exists.
        
        This ensures that if raw sensor data is available for a time slot, the virtual
        sensor calculation is performed for that slot as well.
        """
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="sensor_a",
                    entity_id="sensor.a",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="sensor_b",
                    entity_id="sensor.b",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data for multiple time slots (every 5 minutes for 30 minutes = 6 slots)
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            for i in range(7):  # 0, 5, 10, 15, 20, 25, 30 minutes (7 samples for 6 slots)
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(
                    Sample(
                        entity_id="sensor.a",
                        timestamp=timestamp,
                        value=10.0 + i,  # Varying values
                        unit="unit",
                    )
                )
                session.add(
                    Sample(
                        entity_id="sensor.b",
                        timestamp=timestamp,
                        value=5.0 + i * 0.5,  # Varying values
                        unit="unit",
                    )
                )
            session.commit()

        # Configure a virtual sensor (add operation)
        config = VirtualSensorsConfiguration()
        config.add_sensor(
            VirtualSensorDefinition(
                name="sum_sensor",
                display_name="Sum of A and B",
                description="A + B",
                source_sensor1="sensor_a",
                source_sensor2="sensor_b",
                operation=VirtualSensorOperation.ADD,
                unit="unit",
                enabled=True,
            )
        )
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify that virtual sensor was calculated for all time slots
        with Session(patch_engine_and_config) as session:
            # Count raw sensor samples per category
            sensor_a_samples = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "sensor_a")
                .order_by(ResampledSample.slot_start)
                .all()
            )
            sensor_b_samples = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "sensor_b")
                .order_by(ResampledSample.slot_start)
                .all()
            )
            virtual_samples = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "sum_sensor")
                .order_by(ResampledSample.slot_start)
                .all()
            )
            
            # All should have the same number of samples
            assert len(sensor_a_samples) > 0, "Should have raw samples for sensor_a"
            assert len(sensor_b_samples) > 0, "Should have raw samples for sensor_b"
            assert len(virtual_samples) > 0, "Should have virtual sensor samples"
            
            # Virtual sensor should have samples for EVERY timeframe where raw data exists
            assert len(virtual_samples) == len(sensor_a_samples), (
                f"Virtual sensor should have a sample for each timeframe where raw data exists. "
                f"Expected {len(sensor_a_samples)} virtual samples, got {len(virtual_samples)}"
            )
            assert len(virtual_samples) == len(sensor_b_samples), (
                f"Virtual sensor count should match sensor_b count"
            )
            
            # Verify all have the same time slots
            sensor_a_slots = {s.slot_start for s in sensor_a_samples}
            sensor_b_slots = {s.slot_start for s in sensor_b_samples}
            virtual_slots = {s.slot_start for s in virtual_samples}
            
            assert virtual_slots == sensor_a_slots, "Virtual sensor should have samples for all sensor_a time slots"
            assert virtual_slots == sensor_b_slots, "Virtual sensor should have samples for all sensor_b time slots"
            
            # Verify is_derived flags
            assert all(not s.is_derived for s in sensor_a_samples), "sensor_a should not be derived"
            assert all(not s.is_derived for s in sensor_b_samples), "sensor_b should not be derived"
            assert all(s.is_derived for s in virtual_samples), "Virtual sensor should be derived"
            
            # Verify calculations are correct for each timeframe
            for i, (a_sample, b_sample, v_sample) in enumerate(
                zip(sensor_a_samples, sensor_b_samples, virtual_samples)
            ):
                assert a_sample.slot_start == b_sample.slot_start == v_sample.slot_start, (
                    f"All samples at index {i} should have the same slot_start"
                )
                expected_sum = a_sample.value + b_sample.value
                assert abs(v_sample.value - expected_sum) < 0.01, (
                    f"Virtual sensor calculation wrong at slot {v_sample.slot_start}: "
                    f"expected {expected_sum}, got {v_sample.value}"
                )

    def test_virtual_sensor_not_calculated_when_source_missing(
        self, patch_engine_and_config, temp_config_dir
    ):
        """Test that virtual sensor is NOT calculated for timeframes where source data is missing.
        
        If one of the source sensors doesn't have data for a time slot, the virtual sensor
        should not be calculated for that slot either.
        """
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="sensor_a",
                    entity_id="sensor.a",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="sensor_b",
                    entity_id="sensor.b",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data - sensor_a has data for all time slots, but sensor_b is missing some
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            # sensor_a: samples at 0, 5, 10, 15 minutes (4 slots worth of data)
            for i in range(5):
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(
                    Sample(
                        entity_id="sensor.a",
                        timestamp=timestamp,
                        value=10.0,
                        unit="unit",
                    )
                )
            
            # sensor_b: only samples at 0 and 5 minutes (missing data after that)
            for i in range(2):
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(
                    Sample(
                        entity_id="sensor.b",
                        timestamp=timestamp,
                        value=5.0,
                        unit="unit",
                    )
                )
            session.commit()

        # Configure a virtual sensor
        config = VirtualSensorsConfiguration()
        config.add_sensor(
            VirtualSensorDefinition(
                name="sum_sensor",
                display_name="Sum of A and B",
                description="A + B",
                source_sensor1="sensor_a",
                source_sensor2="sensor_b",
                operation=VirtualSensorOperation.ADD,
                unit="unit",
                enabled=True,
            )
        )
        config.save()

        # Resample
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify behavior
        with Session(patch_engine_and_config) as session:
            # sensor_a should have more samples than sensor_b
            sensor_a_count = session.query(ResampledSample).filter(
                ResampledSample.category == "sensor_a"
            ).count()
            sensor_b_count = session.query(ResampledSample).filter(
                ResampledSample.category == "sensor_b"
            ).count()
            
            # Due to complete-slot semantics, only slots where BOTH sensors have data are saved
            # So we should only have samples for the time slots where both sensors have data
            virtual_count = session.query(ResampledSample).filter(
                ResampledSample.category == "sum_sensor"
            ).count()
            
            # All three should have the same count (only complete slots are saved)
            assert sensor_a_count == sensor_b_count, (
                "Due to complete-slot semantics, sensor_a and sensor_b should have the same count"
            )
            assert virtual_count == sensor_b_count, (
                "Virtual sensor should only be calculated for slots where both sources exist"
            )
            
            # Verify the virtual sensor was calculated for the available slots
            assert virtual_count > 0, "Should have at least one virtual sensor calculation"

    def test_one_hour_twelve_records_scenario(self, patch_engine_and_config, temp_config_dir):
        """Test explicit scenario: 1 hour of data with 5-minute sampling = 12 records per sensor.
        
        If resampling an hour of data (12 five-minute slots), there should be:
        - 12 records for each raw sensor
        - 12 records for each calculated/virtual sensor
        
        This verifies the requirement that virtual sensors are calculated for every timeframe.
        """
        # Create sensor mappings
        with Session(patch_engine_and_config) as session:
            session.add(
                SensorMapping(
                    category="temperature",
                    entity_id="sensor.temperature",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="setpoint",
                    entity_id="sensor.setpoint",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Add sample data for exactly 1 hour (13 samples to cover 12 slots from 10:00 to 11:00)
        # Slots: 10:00-10:05, 10:05-10:10, ..., 10:55-11:00 = 12 slots
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        with Session(patch_engine_and_config) as session:
            for i in range(13):  # 0, 5, 10, 15, ..., 60 minutes (13 samples for 12 slots)
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(
                    Sample(
                        entity_id="sensor.temperature",
                        timestamp=timestamp,
                        value=20.0 + i * 0.1,  # Slowly increasing
                        unit="°C",
                    )
                )
                session.add(
                    Sample(
                        entity_id="sensor.setpoint",
                        timestamp=timestamp,
                        value=21.0,  # Constant
                        unit="°C",
                    )
                )
            session.commit()

        # Configure a virtual sensor (delta = setpoint - temperature)
        config = VirtualSensorsConfiguration()
        config.add_sensor(
            VirtualSensorDefinition(
                name="temp_error",
                display_name="Temperature Error",
                description="Setpoint - Temperature",
                source_sensor1="setpoint",
                source_sensor2="temperature",
                operation=VirtualSensorOperation.SUBTRACT,
                unit="°C",
                enabled=True,
            )
        )
        config.save()

        # Resample with 5-minute intervals
        stats = resample_all_categories(sample_rate_minutes=5)

        # Verify exactly 12 records for each sensor (raw and virtual)
        with Session(patch_engine_and_config) as session:
            temperature_samples = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "temperature")
                .order_by(ResampledSample.slot_start)
                .all()
            )
            setpoint_samples = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "setpoint")
                .order_by(ResampledSample.slot_start)
                .all()
            )
            delta_samples = (
                session.query(ResampledSample)
                .filter(ResampledSample.category == "temp_error")
                .order_by(ResampledSample.slot_start)
                .all()
            )
            
            # Verify exactly 12 records for 1 hour of 5-minute sampling
            assert len(temperature_samples) == 12, (
                f"Expected exactly 12 temperature samples for 1 hour, got {len(temperature_samples)}"
            )
            assert len(setpoint_samples) == 12, (
                f"Expected exactly 12 setpoint samples for 1 hour, got {len(setpoint_samples)}"
            )
            assert len(delta_samples) == 12, (
                f"Expected exactly 12 delta samples for 1 hour, got {len(delta_samples)}. "
                f"Virtual sensors must be calculated for EVERY timeframe where source data exists."
            )
            
            # Verify time slots are correct (10:00, 10:05, 10:10, ..., 10:55)
            expected_times = [base_time + timedelta(minutes=i * 5) for i in range(12)]
            actual_times = [s.slot_start for s in temperature_samples]
            assert actual_times == expected_times, "Time slots should be continuous 5-minute intervals"
            
            # Verify all delta samples exist for the same time slots
            delta_times = [s.slot_start for s in delta_samples]
            assert delta_times == expected_times, "Delta should be calculated for all time slots"
            
            # Verify is_derived flags
            assert all(not s.is_derived for s in temperature_samples), "Raw sensors should not be derived"
            assert all(not s.is_derived for s in setpoint_samples), "Raw sensors should not be derived"
            assert all(s.is_derived for s in delta_samples), "Virtual sensors should be derived"
            
            # Verify delta calculations are correct for each of the 12 slots
            for i, (temp, setpoint, delta) in enumerate(
                zip(temperature_samples, setpoint_samples, delta_samples)
            ):
                expected_delta = setpoint.value - temp.value
                assert abs(delta.value - expected_delta) < 0.01, (
                    f"Slot {i}: Delta calculation incorrect. "
                    f"Expected {expected_delta:.2f}, got {delta.value:.2f}"
                )
                
            # Additional verification: first and last slot
            # First slot (10:00): temp ~20.0, setpoint 21.0, delta ~1.0
            assert delta_samples[0].slot_start == base_time
            assert abs(delta_samples[0].value - 1.0) < 0.15
            
            # Last slot (10:55): temp ~21.2, setpoint 21.0, delta ~-0.2
            assert delta_samples[11].slot_start == base_time + timedelta(minutes=55)
            assert abs(delta_samples[11].value - (-0.2)) < 0.2

