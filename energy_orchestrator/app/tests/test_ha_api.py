"""
Tests for the Home Assistant API helper functions.
"""

import pytest

from ha.ha_api import parse_state_to_float


class TestParseStateToFloat:
    """Tests for the parse_state_to_float function."""

    def test_numeric_string(self):
        """Numeric strings should be converted to float."""
        assert parse_state_to_float("23.5") == 23.5
        assert parse_state_to_float("0") == 0.0
        assert parse_state_to_float("-5.0") == -5.0
        assert parse_state_to_float("100") == 100.0

    def test_on_returns_one(self):
        """'on' should be converted to 1.0."""
        assert parse_state_to_float("on") == 1.0
        assert parse_state_to_float("On") == 1.0
        assert parse_state_to_float("ON") == 1.0

    def test_off_returns_zero(self):
        """'off' should be converted to 0.0."""
        assert parse_state_to_float("off") == 0.0
        assert parse_state_to_float("Off") == 0.0
        assert parse_state_to_float("OFF") == 0.0

    def test_true_returns_one(self):
        """'true' should be converted to 1.0."""
        assert parse_state_to_float("true") == 1.0
        assert parse_state_to_float("True") == 1.0
        assert parse_state_to_float("TRUE") == 1.0

    def test_false_returns_zero(self):
        """'false' should be converted to 0.0."""
        assert parse_state_to_float("false") == 0.0
        assert parse_state_to_float("False") == 0.0
        assert parse_state_to_float("FALSE") == 0.0

    def test_none_returns_none(self):
        """None should return None."""
        assert parse_state_to_float(None) is None

    def test_invalid_string_returns_none(self):
        """Invalid strings should return None."""
        assert parse_state_to_float("unavailable") is None
        assert parse_state_to_float("unknown") is None
        assert parse_state_to_float("") is None
        assert parse_state_to_float("abc") is None

    def test_edge_cases(self):
        """Test edge cases."""
        # Whitespace - should return None (not stripped)
        assert parse_state_to_float(" on") is None
        assert parse_state_to_float("on ") is None
        # Scientific notation should work
        assert parse_state_to_float("1e3") == 1000.0
