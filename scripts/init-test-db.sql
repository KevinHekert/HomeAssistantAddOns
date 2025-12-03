-- Initialize Energy Orchestrator test database
-- This script runs automatically when the MariaDB container starts

USE energy_orchestrator_test;

-- Grant full privileges to the test user
GRANT ALL PRIVILEGES ON energy_orchestrator_test.* TO 'energy_orchestrator'@'%';
FLUSH PRIVILEGES;

-- Create a test verification table to ensure initialization ran
CREATE TABLE IF NOT EXISTS _test_init_marker (
    initialized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO _test_init_marker VALUES (CURRENT_TIMESTAMP);

-- Display initialization confirmation
SELECT 'Database initialized successfully' AS status;
