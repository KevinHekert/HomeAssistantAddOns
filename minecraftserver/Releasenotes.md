# Release Notes

## 1.0.0 - 2025-12-16
- **Server lifecycle**: Ingress UI shows live status while start and restart actions run from the correct Bedrock entrypoint using the configured data directory and runtime files under `/data/run`.
- **World management**: Worlds include immutable names and seeds stored in `/data/worldconfiguration.json`, with new worlds available immediately and existing seeds displayed read-only in the UI.
- **Configuration & access**: Ingress-only access with updated UI warnings for the EULA requirement, configurable data directory support across runtime scripts, and routing that uses relative API paths so status and permissions load reliably.
- **Security**: AppArmor profile allows required helper binaries (e.g., `stdbuf`, `timeout`, `find`, `mkdir`, `rm`, `sleep`) and library access while keeping the add-on restricted.
- **Bundled components**: Ships Bedrock Dedicated Server 1.21.131.1 plus helper tools Easy Add 0.8.11, Entrypoint Demoter 0.4.9, Set-property 0.1.5, Restify 1.7.11, and MC Monitor 0.16.0.
