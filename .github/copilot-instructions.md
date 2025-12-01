# Copilot Instructions for Home Assistant Add-ons Repository

This repository contains Home Assistant add-ons maintained by Kevin Hekert.

## Repository Structure

- **`minecraftserver/`** - Minecraft Bedrock Server add-on for Home Assistant
- **`energy_orchestrator/`** - Energy Orchestrator add-on (work in progress)
- **`repository.json`** - Repository metadata for Home Assistant add-on store
- **`.github/workflows/`** - GitHub Actions for automated updates

## Home Assistant Add-on Development

Each add-on follows the Home Assistant add-on structure:

- `config.yaml` - Add-on configuration and metadata
- `Dockerfile` - Container build instructions
- `build.yaml` - Build arguments and base images
- `README.md` - Documentation for the add-on
- `CHANGELOG.md` - Version history (located within each add-on folder)

### Key Configuration Fields

Add-on `config.yaml` files include:
- `name`, `version`, `slug` - Basic identification
- `arch` - Supported architectures (typically `amd64`, `aarch64`)
- `ingress` - Enable Home Assistant ingress for web UI
- `ingress_port` - Port for the add-on's web interface

## Coding Conventions

- Use semantic versioning for add-on versions
- Document changes in `CHANGELOG.md`
- Follow existing code style in each add-on
- Python code (e.g., Flask apps) should be placed in `app/` directory
- Shell scripts use bash and should be executable

## Testing Requirements

**IMPORTANT: Tests are mandatory for all code changes.**

- For every code change, you MUST update or create corresponding tests
- If a test file does not exist for the changed code, create one
- Test files should be placed in a `tests/` directory within each add-on folder
- Use pytest for Python code testing
- Ensure all tests pass before submitting changes

### Testing Process

1. Identify the code being changed
2. Check if tests exist for that code in `<add-on>/tests/`
3. If tests exist, update them to cover the changes
4. If tests do not exist, create new test files
5. Run all tests to ensure they pass
6. Build the Docker image locally to verify the build succeeds

## Changelog Requirements

**IMPORTANT: Every code change requires a changelog entry.**

- Each add-on has its own `CHANGELOG.md` file located in the add-on folder (e.g., `minecraftserver/CHANGELOG.md`)
- Update the changelog for every change, no matter how small
- Follow the existing changelog format:
  ```
  ## [version] - YYYY-MM-DD
  - Description of change
  ```
- Be descriptive about what changed and why
- Group related changes together under the same version

## Pull Request Requirements

**IMPORTANT: Provide detailed explanations in all pull requests.**

When creating or updating pull requests:

1. **Explain what changed**: Clearly describe all code modifications made
2. **Explain why it changed**: Provide context and reasoning for the changes
3. **Explain what went wrong** (if fixing a bug): Describe the root cause of the issue and how the fix addresses it
4. **List all files modified**: Include a summary of which files were changed and why
5. **Document any breaking changes**: Highlight changes that might affect existing functionality
6. **Include test results**: Show that all tests pass after the changes

### PR Description Template

Use this structure for pull request descriptions:

```
## Summary
Brief overview of the changes.

## Changes Made
- List of specific changes

## Reason for Changes
Explanation of why these changes were necessary.

## Testing
- Description of tests added/updated
- Test results

## Changelog
- Entry added to `<add-on>/CHANGELOG.md`
```

## Automated Workflows

The `bedrock-sync.yml` workflow automatically:
- Checks for Minecraft Bedrock server updates
- Checks for dependency updates (itzg tools)
- Bumps add-on version and updates changelog
- Creates GitHub releases with tags

## Error Handling and Debugging

When something goes wrong:

1. **Identify the error**: Check logs, error messages, and stack traces
2. **Explain the root cause**: Document what caused the issue
3. **Describe the fix**: Explain how the changes resolve the problem
4. **Prevent recurrence**: Add tests to catch similar issues in the future
