"""
Tests for UI structure and organization.

Verifies that:
1. The Configuration tab no longer has the duplicate Sensor Configuration section
2. The Sensor Configuration tab still has all necessary sections
3. Feature Configuration has appropriate guidance text
"""

import pytest
from pathlib import Path


def test_configuration_tab_no_duplicate_sensor_config():
    """Test that the Configuration tab does not have a duplicate Sensor Configuration section."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find the Configuration tab
    config_tab_start = content.find('<div id="tab-configuration" class="tab-content active">')
    assert config_tab_start != -1, "Configuration tab should exist"
    
    # Find the next tab (Sensor Configuration tab)
    sensor_config_tab_start = content.find('<div id="tab-sensorConfig" class="tab-content">')
    assert sensor_config_tab_start != -1, "Sensor Configuration tab should exist"
    assert sensor_config_tab_start > config_tab_start, "Sensor Configuration tab should come after Configuration tab"
    
    # Extract the Configuration tab content (everything between the two tabs)
    config_tab_content = content[config_tab_start:sensor_config_tab_start]
    
    # Verify that the Configuration tab does NOT contain the duplicate sensor configuration section
    assert '<h3>📡 Sensor Configuration</h3>' not in config_tab_content, (
        "Configuration tab should NOT have the duplicate '📡 Sensor Configuration' section"
    )
    
    # Verify that it still has Feature Configuration
    assert '<h3>⚙️ Feature Configuration</h3>' in config_tab_content, (
        "Configuration tab should still have the '⚙️ Feature Configuration' section"
    )


def test_sensor_configuration_tab_has_all_sections():
    """Test that the Sensor Configuration tab has Raw Sensors, Virtual Sensors, and Feature Stats."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find the Sensor Configuration tab
    sensor_config_tab_start = content.find('<div id="tab-sensorConfig" class="tab-content">')
    assert sensor_config_tab_start != -1, "Sensor Configuration tab should exist"
    
    # Find the next tab (training tab)
    training_tab_start = content.find('<div id="tab-training" class="tab-content">')
    assert training_tab_start != -1, "Training tab should exist"
    
    # Extract the Sensor Configuration tab content
    sensor_config_tab_content = content[sensor_config_tab_start:training_tab_start]
    
    # Verify it has all three sections
    assert '<h3>📊 Raw Sensors</h3>' in sensor_config_tab_content, (
        "Sensor Configuration tab should have Raw Sensors section"
    )
    assert '<h3>🔗 Virtual Sensors</h3>' in sensor_config_tab_content, (
        "Sensor Configuration tab should have Virtual Sensors section"
    )
    assert '<h3>📈 Feature Stats Configuration</h3>' in sensor_config_tab_content, (
        "Sensor Configuration tab should have Feature Stats Configuration section"
    )


def test_feature_configuration_has_guidance():
    """Test that Feature Configuration has helpful guidance text."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find Feature Configuration section in Configuration tab
    config_tab_start = content.find('<div id="tab-configuration" class="tab-content active">')
    sensor_config_tab_start = content.find('<div id="tab-sensorConfig" class="tab-content">')
    config_tab_content = content[config_tab_start:sensor_config_tab_start]
    
    # Find Feature Configuration card
    feature_config_start = config_tab_content.find('<h3>⚙️ Feature Configuration</h3>')
    assert feature_config_start != -1, "Feature Configuration section should exist in Configuration tab"
    
    # Extract Feature Configuration card content (from its heading to the next card)
    feature_config_section = config_tab_content[feature_config_start:feature_config_start + 1500]
    
    # Verify it has helpful guidance text
    assert "raw sensors" in feature_config_section.lower(), (
        "Feature Configuration should mention raw sensors"
    )
    assert "virtual" in feature_config_section.lower() or "derived" in feature_config_section.lower(), (
        "Feature Configuration should mention virtual/derived sensors"
    )
    assert "Sensor Configuration" in feature_config_section, (
        "Feature Configuration should reference the Sensor Configuration tab"
    )


def test_tabs_structure():
    """Test that all expected tabs exist in the correct order."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find tab navigation
    tabs_nav_start = content.find('<div class="tabs-nav">')
    assert tabs_nav_start != -1, "Tabs navigation should exist"
    
    # Extract a portion containing tab buttons
    tabs_section = content[tabs_nav_start:tabs_nav_start + 500]
    
    # Verify expected tabs
    assert "⚙️ Configuration" in tabs_section, "Configuration tab button should exist"
    assert "📡 Sensor Configuration" in tabs_section, "Sensor Configuration tab button should exist"
    assert "🤖 Model Training" in tabs_section, "Model Training tab button should exist"
    assert "📊 Sensor Information" in tabs_section, "Sensor Information tab button should exist"


def test_configuration_tab_structure():
    """Test that Configuration tab has expected sections in correct order."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find the Configuration tab
    config_tab_start = content.find('<div id="tab-configuration" class="tab-content active">')
    sensor_config_tab_start = content.find('<div id="tab-sensorConfig" class="tab-content">')
    config_tab_content = content[config_tab_start:sensor_config_tab_start]
    
    # Verify expected sections exist (in order)
    sections = [
        "🔄 Data Resampling",
        "📡 Sync Configuration",
        "⚙️ Feature Configuration",
        "🔀 Two-Step Prediction",
        "🌤️ Weather API Settings",
    ]
    
    last_pos = 0
    for section in sections:
        pos = config_tab_content.find(f'<h3>{section}</h3>', last_pos)
        assert pos != -1, f"Configuration tab should have '{section}' section"
        assert pos > last_pos, f"'{section}' should appear after previous sections"
        last_pos = pos
