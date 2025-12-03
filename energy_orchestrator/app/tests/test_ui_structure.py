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
    assert "model training and prediction" in feature_config_section.lower(), (
        "Feature Configuration should mention model training and prediction"
    )
    assert "special cards" in feature_config_section.lower() or "time/date" in feature_config_section.lower(), (
        "Feature Configuration should mention special cards or time/date features"
    )
    assert "Sensor Configuration" in feature_config_section, (
        "Feature Configuration should reference the Sensor Configuration tab"
    )
    assert "core" in feature_config_section.lower() and "experimental" in feature_config_section.lower(), (
        "Feature Configuration should mention CORE and EXPERIMENTAL features"
    )


def test_tabs_structure():
    """Test that all expected tabs exist in the correct order."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find tab navigation
    tabs_nav_start = content.find('<div class="tabs-nav">')
    assert tabs_nav_start != -1, "Tabs navigation should exist"
    
    # Extract a larger portion containing all tab buttons
    tabs_section = content[tabs_nav_start:tabs_nav_start + 800]
    
    # Verify expected tabs
    assert "⚙️ Configuration" in tabs_section, "Configuration tab button should exist"
    assert "📡 Sensor Configuration" in tabs_section, "Sensor Configuration tab button should exist"
    assert "🤖 Model Training" in tabs_section, "Model Training tab button should exist"
    assert "🔍 Optimizer" in tabs_section, "Optimizer tab button should exist"
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
    # Note: Data Resampling has been moved to Sensor Information tab
    sections = [
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
    
    # Verify Data Resampling is NOT in Configuration tab anymore
    assert "🔄 Data Resampling" not in config_tab_content, (
        "Data Resampling section should have been moved to Sensor Information tab"
    )


def test_loadFeatureConfig_has_no_missing_button_reference():
    """Test that loadFeatureConfig function does not reference non-existent loadFeaturesBtn."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find the loadFeatureConfig function
    func_start = content.find('async function loadFeatureConfig()')
    assert func_start != -1, "loadFeatureConfig function should exist"
    
    # Find the end of the function - look for next async function or end of script
    # Look for the next function definition or closing script tag
    func_end = content.find('async function ', func_start + 10)
    if func_end == -1:
        func_end = content.find('</script>', func_start)
    assert func_end != -1, "Should be able to find end of loadFeatureConfig function"
    
    # Extract the function content
    func_content = content[func_start:func_end]
    
    # Verify the function does NOT reference loadFeaturesBtn
    assert 'loadFeaturesBtn' not in func_content, (
        "loadFeatureConfig should not reference non-existent 'loadFeaturesBtn' button"
    )
    
    # Verify the function does NOT try to manipulate btn.disabled
    assert 'btn.disabled' not in func_content, (
        "loadFeatureConfig should not reference 'btn.disabled' when button doesn't exist"
    )
    
    # Verify the function still gets the status and content divs (it needs these)
    assert 'getElementById("featureConfigStatus")' in func_content or "getElementById('featureConfigStatus')" in func_content, (
        "loadFeatureConfig should still get featureConfigStatus div"
    )
    assert 'getElementById("featureConfigContent")' in func_content or "getElementById('featureConfigContent')" in func_content, (
        "loadFeatureConfig should still get featureConfigContent div"
    )


def test_optimizer_apply_calls_loadFeatureConfig():
    """Test that applying optimizer results correctly calls loadFeatureConfig to refresh UI."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find the applyOptimizerResult function
    func_start = content.find('async function applyOptimizerResult()')
    assert func_start != -1, "applyOptimizerResult function should exist"
    
    # Find the end of the function
    func_end = content.find('async function applyResultById(', func_start)
    assert func_end != -1, "applyResultById function should exist after applyOptimizerResult"
    
    # Extract the function content
    func_content = content[func_start:func_end]
    
    # Verify it calls loadFeatureConfig() after successful apply
    assert 'loadFeatureConfig()' in func_content, (
        "applyOptimizerResult should call loadFeatureConfig() to refresh the feature configuration UI"
    )
    
    # Now check applyResultById as well
    func2_end = content.find('async function loadLatestOptimizerResults()', func_end)
    assert func2_end != -1, "loadLatestOptimizerResults function should exist after applyResultById"
    
    func2_content = content[func_end:func2_end]
    
    # Verify it also calls loadFeatureConfig()
    assert 'loadFeatureConfig()' in func2_content, (
        "applyResultById should also call loadFeatureConfig() to refresh the feature configuration UI"
    )


def test_sensor_information_tab_structure():
    """Test that Sensor Information tab has all expected sections."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find the Sensor Information tab
    sensors_tab_start = content.find('<div id="tab-sensors" class="tab-content">')
    assert sensors_tab_start != -1, "Sensor Information tab should exist"
    
    # Find the end of the tab (look for next closing div at same level or script tag)
    # We'll look for the script tag that comes after all tabs
    script_start = content.find('<script>', sensors_tab_start)
    assert script_start != -1, "Script section should exist"
    
    # Extract the Sensor Information tab content
    sensors_tab_content = content[sensors_tab_start:script_start]
    
    # Verify it has all expected sections
    assert '<h3>📊 Sensor Information</h3>' in sensors_tab_content, (
        "Sensor Information tab should have raw Sensor Information section"
    )
    assert '<h3>📈 Resampled Sensor Information</h3>' in sensors_tab_content, (
        "Sensor Information tab should have Resampled Sensor Information section"
    )
    assert '<h3>🔄 Data Resampling</h3>' in sensors_tab_content, (
        "Sensor Information tab should have Data Resampling section (moved from Configuration tab)"
    )
    
    # Verify the buttons exist
    assert 'id="loadSensorsBtn"' in sensors_tab_content, (
        "Load Sensor Info button should exist"
    )
    assert 'id="loadResampledSensorsBtn"' in sensors_tab_content, (
        "Load Resampled Sensor Info button should exist"
    )
    assert 'id="resampleBtn"' in sensors_tab_content, (
        "Resample Data button should exist"
    )


def test_resampled_sensor_info_function_exists():
    """Test that loadResampledSensorInfo function exists and is properly structured."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Find the loadResampledSensorInfo function
    func_start = content.find('async function loadResampledSensorInfo()')
    assert func_start != -1, "loadResampledSensorInfo function should exist"
    
    # Find the end of the function
    func_end = content.find('async function', func_start + 10)
    assert func_end != -1, "Should be able to find end of function"
    
    # Extract the function content
    func_content = content[func_start:func_end]
    
    # Verify key elements of the function
    assert 'api/resampled_data' in func_content, (
        "loadResampledSensorInfo should call the resampled_data API endpoint"
    )
    assert 'resampledSensorInfoStatus' in func_content, (
        "loadResampledSensorInfo should update the status div"
    )
    assert 'resampledSensorInfoTable' in func_content, (
        "loadResampledSensorInfo should update the table div"
    )
    assert 'is_derived' in func_content, (
        "loadResampledSensorInfo should show is_derived information"
    )


def test_optimizer_best_result_card_removed():
    """Test that the deprecated optimizerBestResultCard has been removed."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    with open(template_path, "r") as f:
        content = f.read()
    
    # Verify the card HTML element does not exist
    assert 'id="optimizerBestResultCard"' not in content, (
        "optimizerBestResultCard should have been removed from the HTML"
    )
    
    # Verify related element IDs are not in the HTML
    assert 'id="optimizerBestConfig"' not in content, (
        "optimizerBestConfig element should not exist"
    )
    assert 'id="optimizerBestModel"' not in content, (
        "optimizerBestModel element should not exist"
    )
    assert 'id="optimizerBestMape"' not in content, (
        "optimizerBestMape element should not exist"
    )
    
    # Verify the runOptimizer function doesn't reference the removed card
    func_start = content.find('async function runOptimizer()')
    assert func_start != -1, "runOptimizer function should exist"
    
    func_end = content.find('async function checkOptimizerStatus()', func_start)
    assert func_end != -1, "checkOptimizerStatus function should exist after runOptimizer"
    
    func_content = content[func_start:func_end]
    
    # The function should not try to manipulate the removed card
    assert 'bestResultCard.style.display' not in func_content, (
        "runOptimizer should not reference bestResultCard"
    )
    assert 'applyBtn.style.display' not in func_content, (
        "runOptimizer should not reference applyBtn from removed card"
    )
