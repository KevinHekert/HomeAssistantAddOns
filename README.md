# Kevin's Home Assistant Add-ons

This repository contains my personal add-ons for Home Assistant. Feel free to install and use the add-ons if you find them useful.

## Installation

Navigate in your Home Assistant frontend to **Settings** -> **Add-ons** -> **Add-on Store** and add this URL as an additional repository:
```txt
https://github.com/KevinHekert/HomeAssistantAddOns
```

## Add-ons in this repository
 - **[Minecraft Server](/minecraftserver/README.md)**: Minecraft server add-on for Home Assistant.
 - **[Energy Orchestrator](/energy_orchestrator/README.md)**: Energy orchestrator add-on for Home Assistant (WIP).

## Development

### Testing

This repository includes comprehensive testing infrastructure:

- **Unit Tests**: Fast tests using SQLite in-memory database
- **Integration Tests**: Complete tests with MariaDB database

#### Running Unit Tests

```bash
cd energy_orchestrator/app
python -m pytest tests/ -v -m "not integration"
```

#### Running Integration Tests

```bash
# Quick start - run all integration tests
./scripts/run-integration-tests.sh

# Run specific test file
./scripts/run-integration-tests.sh test_integration_samples.py

# Keep environment running for debugging
./scripts/run-integration-tests.sh --keep-alive
```

For detailed information about integration testing, see [INTEGRATION_TESTING.md](INTEGRATION_TESTING.md).

### GitHub Actions

- **Unit Tests**: Run automatically on every push and PR
- **Integration Tests**: Run automatically on changes to Energy Orchestrator
- **Bedrock Sync**: Automated Minecraft server version updates

See `.github/workflows/` for workflow configurations.
