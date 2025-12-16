## 1.2.30 - 2025-12-16
- Updated Bedrock Server from '1.21.130.4' to '1.21.131.1'

## 1.2.29 - 2025-12-22
- Permit `/usr/bin/rm` in AppArmor profile so startup cleanup can run.

## 1.2.28 - 2025-12-21
- Version bump for maintenance release.

## 1.2.27 - 2025-12-20
- Fixed saving seeds for newly created and existing worlds without a stored seed.
- Allow filling in missing seeds so they are persisted to the world configuration.
- Ensure new worlds and their seeds become available immediately in the selection list.

## 1.2.26 - 2025-12-16
- Updated Bedrock Server from '1.21.130.4' to preview build '1.26.0.25' (bin-linux-preview)

## 1.2.25 - 2025-12-16
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
