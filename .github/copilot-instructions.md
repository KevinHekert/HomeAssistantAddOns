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
- `CHANGELOG.md` - Version history

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

## Automated Workflows

The `bedrock-sync.yml` workflow automatically:
- Checks for Minecraft Bedrock server updates
- Checks for dependency updates (itzg tools)
- Bumps add-on version and updates changelog
- Creates GitHub releases with tags

## Testing Changes

Add-ons are built and tested through Home Assistant's add-on build system. Test changes by:
1. Building the Docker image locally
2. Installing the add-on from your development repository in Home Assistant
