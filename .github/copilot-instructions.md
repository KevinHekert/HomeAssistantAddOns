# GitHub Copilot Instructions

This repository contains Home Assistant Add-ons. Below are the guidelines and conventions for developing and maintaining the add-ons in this repository.

## Development Requirements

**Always test everything and write tests when developing.** This is a mandatory requirement for all code changes:

- Run all existing tests before and after making changes
- Write new tests for any new functionality or bug fixes
- Ensure all tests pass before considering a change complete
- Follow test-driven development (TDD) practices when appropriate

**Always verify builds and runtime functionality:**

- Always check that the code builds successfully
- Always verify that the application runs correctly after changes
- Test the actual functionality, not just unit tests

**Always run tests automatically and report results:**

- Run all tests automatically as part of the development process
- Post test results in the pull request conversation
- Include both passed and failed test summaries

**Always explain changes in pull requests:**

- Provide a clear functional explanation of what was changed and why
- Describe the impact of changes on the application behavior
- Document any breaking changes or migration steps required

## Repository Structure

```
/
├── energy_orchestrator/    # Energy Orchestrator add-on (Flask-based, Python)
├── minecraftserver/        # Minecraft Bedrock Server add-on (Bash scripts, Python Flask UI)
├── repository.json         # Home Assistant add-on repository metadata
└── README.md               # Repository documentation
```

## Add-on Structure

Each add-on follows the standard Home Assistant Add-on structure:

- `config.yaml` - Add-on configuration and metadata
- `Dockerfile` - Container build instructions
- `build.yaml` - Build arguments and base images (if needed)
- `README.md` - Add-on documentation
- `CHANGELOG.md` - Version history

## Coding Conventions

### Python

- Use Python 3 with Flask for web interfaces
- Use type hints where appropriate
- Follow PEP 8 style guidelines
- Use logging module for application logs
- For the Energy Orchestrator add-on:
  - Flask app is in `energy_orchestrator/app/`
  - Tests are in `energy_orchestrator/app/tests/`
  - Dependencies are in `requirements.txt`

### Shell Scripts

- Use `#!/usr/bin/with-contenv bashio` for Home Assistant add-on entry scripts (the bashio framework handles error management)
- Use `#!/bin/bash` with `set -e` for standalone utility scripts
- Use `bashio::log.info` for logging in Home Assistant add-on scripts
- Use `bashio::config` to read add-on configuration values

### Docker

- Use multi-stage builds where appropriate to reduce image size
- Use Alpine-based images when possible (`ghcr.io/home-assistant/*-base`)
- For Debian-based requirements, use `ghcr.io/home-assistant/*-base-debian`
- Install only necessary dependencies
- Clean up package caches after installation

### Configuration

- Define add-on options in `config.yaml` under `options:` and `schema:`
- Use appropriate schema types: `str`, `str?` (optional), `int`, `bool`, etc.
- Provide sensible default values

## Testing

### Energy Orchestrator

The Energy Orchestrator add-on has Python tests located in `energy_orchestrator/app/tests/`. Run tests with:

```bash
cd energy_orchestrator/app
python -m pytest tests/
```

## Build and Deployment

- Add-ons are built and deployed through Home Assistant's add-on build system
- The Minecraft Server add-on uses automated version updates via GitHub Actions (`.github/workflows/bedrock-sync.yml`)

## Important Notes

- All add-ons run inside Docker containers managed by Home Assistant
- Use `ingress: true` in `config.yaml` for web UIs accessible through Home Assistant
- For network services, consider using `host_network: true` when port mapping is problematic
- Always handle configuration errors gracefully and provide meaningful error messages

## Languages

- Code comments may be in Dutch or English (both are acceptable in this repository)
- User-facing documentation should be in English
