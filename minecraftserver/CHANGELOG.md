## 1.0.4 - 2026-02-09
- Updated MC_MONITOR from '0.16.0' to '0.16.1'

## 1.0.3 - 2026-02-08
- Updated RESTIFY from '1.7.11' to '1.7.12'

## 1.0.2 - 2026-01-09
- Updated Bedrock Server from '1.21.132.1' to '1.21.132.3'

## 1.0.1 - 2026-01-08
- Updated Bedrock Server from '1.21.131.1' to '1.21.132.1'

### Functionaliteit (toegevoegd na 1.0.0)
- Addon met Minecraft Bedrock Dedicated Server
- Correcte runtime-afhandeling met juiste working directory en data path
- Health checks en procesbewaking (MC Monitor)
- Detectie en selectie van meerdere werelden (niet simultaan)
- World switching via configuratie
- World-specifieke seeds
- Opslag van seeds per wereld in `/data/worldconfiguration.json`
- Automatisch opslaan van ontbrekende seeds bij bestaande werelden
- Nieuwe werelden en seeds direct beschikbaar in de UI
- Worldnaam en seed zijn immutable na aanmaak
- Logging van worldnaam en seed bij serverstart
- EULA-afhandeling via configuratie (default `false`)
- Server start niet zonder expliciete EULA-acceptatie
- Config-gedreven world detectie en selectie
- Volledige bediening via Home Assistant Ingress
- Actief AppArmor-profiel voor de add-on
- Toegestane helper binaries (o.a. `stdbuf`, `timeout`, `find`, `sleep`, `mkdir`, `rm`, `tail`)
- Toegang tot `/usr/libexec` voor stdbuf/preload-detectie
- Runtime-bestanden uitsluitend in schrijfbare directories
- Add-on uitsluitend toegankelijk via Ingress (deny-all, ingress-only)
- Gecontroleerde log-output zonder LD_PRELOAD-warnings
- Fallback logging indien `stdbuf` niet beschikbaar is
- Watchdog-loop voor stabiele procesbewaking

Package met (allen geleverd door https://github.com/itzg)
- Entrypoint Demoter
- MC Monitor
- Set-property
- Restify
- Easy Add
- GitHub Actions voor automatische Bedrock-versiedetectie en update
- Automatische update van build parameters (`build.yaml`)
- Gesynchroniseerd add-on versiebeheer
- Logo en icon toegevoegd
- Betrouwbare update-flow voor de add-on
- Onderhoudsreleases zonder handmatige migratie
