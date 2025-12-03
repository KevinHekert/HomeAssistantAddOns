"""
Tests for optimizer configuration functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from app import app
from db.optimizer_config import get_optimizer_config, set_optimizer_config


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestOptimizerConfig:
    """Test optimizer configuration storage and retrieval."""
    
    def test_get_optimizer_config_returns_defaults(self):
        """Get optimizer config returns default values when no config exists."""
        with patch('db.optimizer_config.Session') as mock_session:
            # Mock empty result (no config in database)
            mock_session.return_value.__enter__.return_value.scalars.return_value.first.return_value = None
            
            config = get_optimizer_config()
            
            assert isinstance(config, dict)
            assert "max_workers" in config
            assert config["max_workers"] is None  # Default is auto-calculate
    
    def test_get_optimizer_config_returns_stored_value(self):
        """Get optimizer config returns stored max_workers value."""
        with patch('db.optimizer_config.Session') as mock_session:
            # Mock config with max_workers = 5
            mock_config = MagicMock()
            mock_config.max_workers = 5
            mock_session.return_value.__enter__.return_value.scalars.return_value.first.return_value = mock_config
            
            config = get_optimizer_config()
            
            assert config["max_workers"] == 5
    
    def test_set_optimizer_config_creates_new_config(self):
        """Set optimizer config creates new config when none exists."""
        with patch('db.optimizer_config.Session') as mock_session:
            mock_sess = mock_session.return_value.__enter__.return_value
            # Mock no existing config
            mock_sess.scalars.return_value.first.return_value = None
            
            result = set_optimizer_config(max_workers=3)
            
            assert result is True
            mock_sess.add.assert_called_once()
            mock_sess.commit.assert_called_once()
    
    def test_set_optimizer_config_updates_existing_config(self):
        """Set optimizer config updates existing config."""
        with patch('db.optimizer_config.Session') as mock_session:
            mock_sess = mock_session.return_value.__enter__.return_value
            # Mock existing config
            mock_config = MagicMock()
            mock_config.max_workers = 2
            mock_sess.scalars.return_value.first.return_value = mock_config
            
            result = set_optimizer_config(max_workers=7)
            
            assert result is True
            assert mock_config.max_workers == 7
            mock_sess.commit.assert_called_once()
    
    def test_set_optimizer_config_converts_zero_to_none(self):
        """Set optimizer config converts 0 to None for consistency."""
        with patch('db.optimizer_config.Session') as mock_session:
            mock_sess = mock_session.return_value.__enter__.return_value
            mock_config = MagicMock()
            mock_sess.scalars.return_value.first.return_value = mock_config
            
            result = set_optimizer_config(max_workers=0)
            
            assert result is True
            assert mock_config.max_workers is None
    
    def test_set_optimizer_config_rejects_negative_value(self):
        """Set optimizer config rejects negative max_workers."""
        result = set_optimizer_config(max_workers=-1)
        
        assert result is False


class TestOptimizerConfigAPI:
    """Test optimizer configuration API endpoints."""
    
    def test_get_optimizer_config_endpoint(self, client):
        """GET /api/optimizer/config returns configuration."""
        with patch('app.get_optimizer_config') as mock_get_config:
            mock_get_config.return_value = {"max_workers": 4}
            
            response = client.get('/api/optimizer/config')
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["config"]["max_workers"] == 4
    
    def test_set_optimizer_config_endpoint_valid_value(self, client):
        """POST /api/optimizer/config with valid max_workers."""
        with patch('app.set_optimizer_config') as mock_set_config:
            mock_set_config.return_value = True
            
            response = client.post('/api/optimizer/config', 
                                   json={"max_workers": 6})
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            mock_set_config.assert_called_once_with(max_workers=6)
    
    def test_set_optimizer_config_endpoint_null_value(self, client):
        """POST /api/optimizer/config with null max_workers."""
        with patch('app.set_optimizer_config') as mock_set_config:
            mock_set_config.return_value = True
            
            response = client.post('/api/optimizer/config', 
                                   json={"max_workers": None})
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            mock_set_config.assert_called_once_with(max_workers=None)
    
    def test_set_optimizer_config_endpoint_zero_value(self, client):
        """POST /api/optimizer/config with zero max_workers (auto-calculate)."""
        with patch('app.set_optimizer_config') as mock_set_config:
            mock_set_config.return_value = True
            
            response = client.post('/api/optimizer/config', 
                                   json={"max_workers": 0})
            
            assert response.status_code == 200
            mock_set_config.assert_called_once_with(max_workers=0)
    
    def test_set_optimizer_config_endpoint_negative_value(self, client):
        """POST /api/optimizer/config rejects negative max_workers."""
        response = client.post('/api/optimizer/config', 
                               json={"max_workers": -5})
        
        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
    
    def test_set_optimizer_config_endpoint_invalid_type(self, client):
        """POST /api/optimizer/config rejects non-integer max_workers."""
        response = client.post('/api/optimizer/config', 
                               json={"max_workers": "abc"})
        
        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
    
    def test_set_optimizer_config_endpoint_no_data(self, client):
        """POST /api/optimizer/config rejects empty request."""
        response = client.post('/api/optimizer/config', 
                               headers={'Content-Type': 'application/json'})
        
        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
