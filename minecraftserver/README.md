# Minecraft Bedrock Server Add-on for Home Assistant

Run your own **Minecraft Bedrock Server** directly inside Home Assistant – with support for automatic updates, multiple worlds, MCPack installation, and full configuration through the HA UI.

This add-on is designed for **simplicity**, **performance**, and **zero-maintenance hosting** for families, kids, or LAN-based multiplayer.

**Important** EULA setting must be set to true in order to start server.

---

## Credits & acknowledgements

This add-on builds upon several excellent open-source tools.  
Full credit and appreciation go to the original authors and maintainers:

| Tool | Purpose | Author / Project |
|-----|--------|------------------|
| **MC Monitor** | Health checks and server availability monitoring | [itzg](https://github.com/itzg/) |
| **Entrypoint Demoter** | Secure privilege dropping for container entrypoints | [itzg](https://github.com/itzg/) |
| **Set-property** | Safe and declarative configuration file updates | [itzg](https://github.com/itzg/) |
| **Restify** | Lightweight REST API helper for runtime control | [itzg](https://github.com/itzg/) |
| **Easy Add** | Simplified add-on runtime utilities | [itzg](https://github.com/itzg/) |


## ✨ Features

| Feature | Description |
|:--------|:--------------|
| ✅ Bedrock Dedicated Server | Runs the official Mojang Bedrock Server inside Home Assistant |
| 🔄 Auto-version detection | Repository is updated automatically with the latest Bedrock version |
| 🌍 World configuration | Seeds, world type, difficulty, gamemode & cheats |
| 👥 Player management | Whitelist/allowlist, default permissions, max player limit |
| 🚀 Performance tuning | View distance, tick distance, multithreading, compression |
| 🧱 Anti-cheat controls | Server authoritative movement & validation thresholds |
| 🧠 Easy UI | Configuration through HA UI with friendly labels & dropdowns |
| 🌐 LAN visibility toggle | Enable/disable server broadcast on local network |
| 🧑‍💻 Works with host network | No port mapping headaches — plug & play |

## 🏗 Installation

1. Add this repository to Home Assistant Add-on Store: `https://github.com/KevinHekert/HomeAssistantAddOns/`
2. Install **Minecraft Server** add-on  
3. Open Configuration tab  
4. Adjust settings as needed  
5. Start the add-on  

> First startup may take longer due to version download.

---

## ⚙️ Configuration

All settings can be modified via the Home Assistant Add-on UI.

The configuration is grouped into 5 logical sections:

### 🧩 General
Core server behavior such as name, ports, online mode, and LAN visibility.

| Setting | Description |
|:--------|:------------|
| Display name | Name shown in the Minecraft server list |
| Port (IPv4/IPv6) | Ports used for Bedrock server |
| Transport | `nethernet` (default) or `raknet` |
| Online verification | Require Microsoft/Xbox authentication |
| LAN visibility | Broadcast server on LAN |
| Send telemetry | Allow usage analytics to Microsoft |

> **Transport:** Bedrock 1.26.51.1 and newer only accept players over NetherNet.
> On `raknet` the server still starts and answers pings, but every join attempt
> fails and the log prints `TRANSPORT TYPE ERROR`. NetherNet turns the IPv4 port
> into a TCP signalling socket and negotiates gameplay over UDP per client, so
> LAN auto-discovery may not list the server — add it by address instead.

---

### 🌍 World
Controls how your world looks, feels, and behaves.

| Setting | Options |
|:--------|:--------|
| World name | Any text |
| World seed | Number or text seed |
| World type | Default / Flat / Legacy |
| Game mode | Survival / Creative / Adventure |
| Difficulty | Peaceful / Easy / Normal / Hard |
| Allow cheats | Yes/No |

---

### 👥 Players

| Setting | Description |
|:--------|:------------|
| Max players | Maximum number of concurrent players |
| Whitelist / Allowlist | Restrict players |
| Default permission | Visitor / Member / Operator |
| Require texture pack | Force clients to install server pack |

---

### 🚀 Performance

Tweak simulation radius, CPU usage & bandwidth.

| Setting | Description |
|:--------|:----------------|
| View distance | Render distance for players |
| Simulation distance | Radius where mobs & redstone tick |
| Idle timeout | Kick inactive players |
| Max threads | 0 = auto |
| Compression threshold | Lower uses more CPU, less bandwidth |

---

### 🛡 Anti-Cheat & Movement

Server authoritative movement ensures fair gameplay and prevents hacked clients.

| Setting | Options |
|:--------|:------------|
| Movement authority | Client / Server / Server+Rewind |
| Block breaking validation | Validate block break actions |
| Score threshold | Cheating sensitivity |
| Distance & duration thresholds | Movement tolerance |
| Correct movement | Fixes illegal movement via teleport |

---

