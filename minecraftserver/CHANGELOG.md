## 1.3.7 - 2025-12-16
- Simplified web UI API calls to use direct relative paths and removed the unused ingress API helper.
- Use relative API URLs (without a leading slash) for server control requests so ingress keeps working without custom path handling.
- Use relative ingress API paths for runtime permissions so permissions data loads inside Home Assistant.
- Allow AppArmor to read libraries under `/usr/libexec` so the stdbuf detection logic can access alternate preload paths.
- Fix ingress API calls so server status and control buttons work correctly in the web UI.
- Avoid LD_PRELOAD warnings when `stdbuf` support is unavailable by falling back to direct log filtering.
- Allow additional helper binaries (e.g., `stdbuf`, `timeout`, `find`) in the AppArmor profile to prevent permission errors during logging and health checks.
- Allowed `/usr/bin/sleep` in the AppArmor profile so the watchdog loop in `/opt/start.sh` can wait without permission errors.
- Consolidated the changes from the 1.3.x line under this version number.
- Allowed `/usr/bin/mkdir` in the AppArmor profile to prevent permission errors when creating the runtime directory during startup.
- Direct Bedrock runtime files (PID and stop markers) to the writable `/data/run` directory for reliable startup and control scripts.
- Permit `/usr/bin/rm` in AppArmor profile so startup cleanup can run.
- Fixed saving seeds for newly created and existing worlds without a stored seed, allowing missing seeds to be persisted to the world configuration.
- Ensure new worlds and their seeds become available immediately in the selection list.
- Version bump for maintenance release.
- Added world-specific seed configuration
- Seeds are now saved per world in /data/worldconfiguration.json
- World names and seeds are immutable once created
- Added logging of world name and seed on startup
- Seed field is read-only for existing worlds in UI

## 1.2.24 - 2025-12-09
- Updated Bedrock Server from '1.21.124.2' to '1.21.130.4'

## 1.2.23 - 2025-11-25
- Removed depricated code, adjusted comments in code.

## 1.2.22 - 2025-11-21
- Updated Bedrock Server from '1.21.123.2' to '1.21.124.2'
- Updated GitHub action to edit version in build parameters (build.yaml)

## 1.2.8 - 2025-11-20
- Rephrased items; updating isn't automated. Version detection is, within GitHub: but it isn't in the code itself. 
- Added logo and icon
- Corrected logo ratio (1.2.7 -> 1.2.8)

## 1.2.6 - 2025-11-19
- Added warning to UI, server won't start without EULA.
- Add tail to apparmor
- Bugfix: unable to set EULA=True if EULA=False
- Move EULA to config, default = false. 
- Whitelist depricated, allowlist is used. 
- Use permissions to fill allowlist. 
- New UI for config!
- World detection from config, support for world switching
- Apparmor added to addon
- Ingress added to addon
- Add-on access only via Ingress deny all, access only via Ingriss


## 1.0.7-14 - 2025-11-14
- Fixed update flow
- Updated ENTRYPOINT_DEMOTER from '0.4.8' to '0.4.9'
- Changed naming to final. No first name references.

## 1.0.6 - 2025-11-13
- New version Bedrock Server package available: 1.21.123.2
- Updated installation parameters in `minecraftserver/build.yaml` (BDS_URL & BDS_VERSION)

## 1.0.5 - 2025-11-10
- Updated Entrypoint Demoter to latest stable `0.4.9`
- Updated MC Monitor to latest stable `0.15.8`
- Updated Set-property to latest stable `0.1.5`
- Updated Restify to latest stable `1.7.11`
- Updated Easy Add to latest stable `0.8.11`

## 1.0.4 - 2025-11-10
- New version Bedrock Server package installed: 1.21.122.2
- Updated `minecraftserver/build.yaml` (BDS_URL & BDS_VERSION)
- Bumped add-on version in `minecraftserver/config.yaml` to 1.0.4

## 1.0.3 - 2025-11-08
- Added url to config

## 1.0.2 - 2025-11-07
- Changed slug to final version

## 1.0.1 - 2025-11-07
- New version Bedrock Server package installed: 1.21.121.1
- Updated `minecraftserver/build.yaml` (BDS_URL & BDS_VERSION)
- Bumped add-on version in `minecraftserver/config.yaml` to 1.0.2

## 1.0.0 - 2025-11-07
- Initial release
