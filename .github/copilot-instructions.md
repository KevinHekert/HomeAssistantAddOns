# GitHub Copilot Instructions for Home Assistant Add-ons Repository

This repository contains Home Assistant add-ons maintained by Kevin Hekert. Below are the guidelines and conventions for developing and maintaining the add-ons in this repository.

---

## Development Requirements

**Tests are mandatory for all code changes. Always test everything and write tests when developing.**

- Run all existing tests before and after making changes.
- Write new tests for any new functionality or bug fixes.
- Ensure all tests pass before considering a change complete.
- Follow test-driven development (TDD) practices when appropriate.

**Always verify builds and runtime functionality:**

- Check that the code builds successfully (Docker image must build).
- Verify that the add-on runs correctly after changes.
- Test real functionality, not just unit tests.

**Always run tests automatically and report results:**

- Run all tests as part of the development process (locally and/or in CI).
- Post test results in the pull request description.
- Include both passed and failed test summaries (if something fails, explain why).

**Always explain changes in pull requests:**

- Provide a clear functional explanation of what was changed and why.
- Describe the impact of changes on add-on behavior.
- Document any breaking changes or migration steps required.

---

## Repository Structure

Top-level structure:

- `minecraftserver/` – Minecraft Bedrock Server add-on for Home Assistant.
- `energy_orchestrator/` – Energy Orchestrator add-on (Flask-based, Python, work in progress).
- `repository.json` – Repository metadata for the Home Assistant add-on store.
- `README.md` – Repository-level documentation.
- `.github/workflows/` – GitHub Actions for automated builds and updates (e.g. Bedrock sync).

---

## Add-on Structure

Each add-on follows the standard Home Assistant add-on structure:

- `config.yaml` – Add-on configuration and metadata.
- `Dockerfile` – Container build instructions.
- `build.yaml` – Build arguments and base images (if needed).
- `README.md` – Add-on documentation.
- `CHANGELOG.md` – Version history (per add-on).

### Key Configuration Fields (`config.yaml`)

Typical fields:

- `name`, `version`, `slug` – Basic identification.
- `arch` – Supported architectures (typically `amd64`, `aarch64`).
- `startup`, `boot`, `host_network`, `host_ipc`, etc. – Add-on runtime options.
- `ingress` – Enable Home Assistant ingress for web UIs.
- `ingress_port` – Port for the add-on’s web interface (if needed).
- `options` – Default configuration values.
- `schema` – Validation schema for options (`str`, `str?`, `int`, `bool`, etc.).

---

## Coding Conventions

### General

- Use semantic versioning for add-on versions.
- Follow the existing style and structure in each add-on.
- Keep changes small, focused, and well-documented.
- Prefer readability and maintainability over cleverness.

### Python

- Use Python 3 for all Python code.
- Use Flask for web interfaces (where applicable).
- Use type hints where appropriate.
- Follow PEP 8 style guidelines.
- Use the `logging` module for application logs.
- Keep Python application code in an `app/` directory where possible.

For the Energy Orchestrator add-on:

- Flask app is located in: `energy_orchestrator/app/`
- Tests are located in: `energy_orchestrator/app/tests/`
- Dependencies are defined in: `energy_orchestrator/requirements.txt` (or similar requirements file)

### Shell Scripts

- For Home Assistant add-on entry scripts:
  - Use `#!/usr/bin/with-contenv bashio`
  - Use `bashio::log.info` (and related functions) for logging.
  - Use `bashio::config` to read add-on configuration values.
- For standalone utility scripts:
  - Use `#!/bin/bash`
  - Use `set -e` (and optionally `set -u`, `set -o pipefail`) for safer scripts.
- Keep scripts idempotent where possible and fail fast with clear error messages.

### Docker

- Use multi-stage builds when appropriate to reduce image size.
- Prefer Home Assistant base images:
  - Alpine-based: `ghcr.io/home-assistant/*-base`
  - Debian-based: `ghcr.io/home-assistant/*-base-debian` when Debian is required.
- Install only necessary dependencies.
- Clean up package caches and temporary files after installation.
- Avoid running processes as root inside containers where feasible.

### Configuration

- Define all add-on options under `options:` and `schema:` in `config.yaml`.
- Use appropriate schema types (`str`, `str?`, `int`, `bool`, etc.).
- Provide sensible default values.
- Validate user input and handle configuration errors gracefully.

---

## Testing

**Testing is required for every code change.**

### General Testing Rules

- For every code change, update or create corresponding tests.
- If a test file does not exist for the changed code, create one.
- Place test files in a `tests/` directory within each add-on folder (e.g. `minecraftserver/tests/`, `energy_orchestrator/app/tests/`).
- Use `pytest` for testing Python code.
- Ensure all tests pass before submitting or merging a pull request.

### Energy Orchestrator Add-on

Tests for the Energy Orchestrator add-on are located in `energy_orchestrator/app/tests/`.

Run tests with:

    cd energy_orchestrator/app
    python -m pytest tests/

---

## Changelog Requirements

**Every code change requires a changelog entry.**

- Each add-on has its own `CHANGELOG.md` in that add-on’s folder (e.g. `minecraftserver/CHANGELOG.md`).
- Update the changelog for every change, regardless of size.
- Follow the existing changelog format. Example:

    ## [1.2.3] - 2025-12-01
    - Short, clear description of the change
    - Additional details if needed

- Be descriptive about what changed and why.
- Group related changes together under the same version.

---

## Pull Request Requirements

**Pull requests must be detailed and self-explanatory.**

When creating or updating pull requests, include at least:

1. Summary  
   - Brief overview of the changes.

2. Changes Made  
   - Bullet list of specific changes (files / components / behavior).

3. Reason for Changes  
   - Why these changes were necessary (feature, refactor, bug fix, etc.).

4. Testing  
   - Description of tests added or updated.
   - How tests were executed (e.g. pytest, Docker build).
   - Confirmation that all tests pass.

5. Changelog  
   - Confirmation that a changelog entry was added to `<add-on>/CHANGELOG.md`.

6. Breaking Changes (if any)  
   - Explicitly list anything that might break existing setups.
   - Describe required migration steps.

### Example PR Description Template

    ## Summary
    Brief overview of the changes.

    ## Changes Made
    - List of specific changes

    ## Reason for Changes
    Explanation of why these changes were necessary.

    ## Testing
    - Tests added/updated:
      - ...
    - Test results:
      - All tests passed locally
      - CI status: ...

    ## Changelog
    - Entry added to `<add-on>/CHANGELOG.md`: [version] - YYYY-MM-DD

---

## Build and Deployment

- Add-ons are built and deployed through Home Assistant’s add-on build system.
- Docker images must build without errors before merging changes.
- Ensure that add-on metadata (`config.yaml`, `repository.json`) is consistent and valid.

### Automated Workflows

The GitHub Actions workflow `bedrock-sync.yml`:

- Checks for Minecraft Bedrock server updates.
- Checks for dependency updates (e.g. itzg tools).
- Bumps the Minecraft add-on version and updates the changelog.
- Creates GitHub releases with tags.

Do not break these workflows. If you modify workflows:

- Explain the reason in the PR.
- Test the workflow changes where possible (e.g. via `act` or dry runs).
- Ensure they remain compatible with the existing repository structure.

---

## Error Handling and Debugging

When something goes wrong:

1. Identify the error  
   - Check logs, error messages, and stack traces.

2. Explain the root cause  
   - In code comments and/or PR description, document what caused the issue.

3. Describe the fix  
   - Explain how the change resolves the issue.
   - Reference relevant code blocks or files.

4. Prevent recurrence  
   - Add or update tests to catch similar issues in the future.
   - Where possible, add validation or safeguards in the code.

---

## Languages

- Code comments can be in Dutch or English (both are acceptable).
- User-facing documentation (README, logs shown to end-users, UI texts) must be in English.

---