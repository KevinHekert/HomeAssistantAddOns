ARG TARGETARCH
FROM ghcr.io/home-assistant/amd64-base-debian:trixie

# Multi-arch variabelen
ARG TARGETOS
ARG TARGETARCH
ARG TARGETPLATFORM

# ===== build args voor Bedrock versie/bron =====
ARG BDS_VERSION
ARG BDS_URL
ARG BDS_SHA256

# ===== build args voor extra tools =====
ARG EASY_ADD_VERSION
ARG ENTRYPOINT_DEMOTER_VERSION
ARG SET_PROPERTY_VERSION
ARG RESTIFY_VERSION
ARG MC_MONITOR_VERSION


# Tools installeren en up to date maken
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y curl unzip jq
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-flask python3-waitress
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Instal box64 on arm
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      echo "🧱 Installing Box64 from Debian repository (arm64)..."; \
      set -eux; \
      apt-get update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends box64; \
      apt-get clean; \
      rm -rf /var/lib/apt/lists/*; \
    else \
      echo "Skipping Box64 installation for $TARGETARCH"; \
    fi

RUN echo "🏗️ Building for platform: ${TARGETPLATFORM} (OS=${TARGETOS}, ARCH=${TARGETARCH})"
RUN echo "🧱 Bedrock Server version: ${BDS_VERSION}"
RUN echo "🌐 Bedrock Server URL: ${BDS_URL}"

# Default bedrock poort openen.
EXPOSE 19132/udp 8789/tcp

VOLUME ["/data"]
WORKDIR /data

#ENTRYPOINT ["/usr/local/bin/entrypoint-demoter", "--match", "/data", "--debug", "--stdin-on-term", "stop", "/opt/bedrock-entry.sh"]
ENTRYPOINT ["/usr/local/bin/entrypoint-demoter", "--match", "/data", "--debug", "--stdin-on-term", "stop", "/opt/start.sh"]


ARG EASY_ADD_VERSION

ADD https://github.com/itzg/easy-add/releases/download/${EASY_ADD_VERSION}/easy-add_linux_${TARGETARCH} /usr/local/bin/easy-add
RUN chmod +x /usr/local/bin/easy-add

RUN easy-add --var version=${ENTRYPOINT_DEMOTER_VERSION} --var app=entrypoint-demoter --file {{.app}} --from https://github.com/itzg/{{.app}}/releases/download/v{{.version}}/{{.app}}_{{.version}}_linux_${TARGETARCH}.tar.gz
RUN easy-add --var version=${SET_PROPERTY_VERSION} --var app=set-property --file {{.app}} --from https://github.com/itzg/{{.app}}/releases/download/{{.version}}/{{.app}}_{{.version}}_linux_${TARGETARCH}.tar.gz
RUN easy-add --var version=${RESTIFY_VERSION} --var app=restify --file {{.app}} --from https://github.com/itzg/{{.app}}/releases/download/{{.version}}/{{.app}}_{{.version}}_linux_${TARGETARCH}.tar.gz
RUN easy-add --var version=${MC_MONITOR_VERSION} --var app=mc-monitor --file {{.app}} --from https://github.com/itzg/{{.app}}/releases/download/{{.version}}/{{.app}}_{{.version}}_linux_${TARGETARCH}.tar.gz

# Bestanden naar container kopiëren
COPY bedrock-entry.sh /opt/bedrock-entry.sh
COPY start.sh /opt/start.sh
COPY healthcheck.sh /opt/healthcheck.sh
COPY *.mcpack /opt/addons/
COPY property-definitions.json /etc/bds-property-definitions.json
COPY web/app.py /opt/flask/app.py
COPY web/static /opt/flask/static
COPY bin/* /usr/local/bin/

# Maak scripts uitvoerbaar
RUN chmod +x /opt/bedrock-entry.sh
RUN chmod +x /opt/start.sh
RUN chmod +x /opt/healthcheck.sh


# ===== Download & uitpakken tijdens build =====
# (1) Downloaden
RUN set -eux;
RUN test -n "${BDS_VERSION}" && test -n "${BDS_URL}"
RUN echo "📦 Downloading Bedrock Server version ${BDS_VERSION}..."
RUN curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors --http1.1 -A "itzg/minecraft-bedrock-server" -o /tmp/bedrock.zip "${BDS_URL}"


# (2) Uitpakken naar /opt/bds (read-only runtime, data in /data)
RUN mkdir -p /opt/bds
WORKDIR /opt/bds
RUN unzip -q /tmp/bedrock.zip && rm -f /tmp/bedrock.zip


# (3) Hernoem binary naar vaste naam met versie-suffix
RUN chmod +x bedrock_server && mv bedrock_server "bedrock_server-${BDS_VERSION}"


# (4) Bewaar versie-info voor runtime
RUN echo "${BDS_VERSION}" > /etc/bds-version
RUN mkdir -p /data/worlds
RUN ln -sfn /data/worlds            /opt/bds/worlds && \
    ln -sfn /data/server.properties /opt/bds/server.properties && \
    ln -sfn /data/allowlist.json    /opt/bds/allowlist.json && \
    ln -sfn /data/permissions.json  /opt/bds/permissions.json
    
ENV VERSION=${BDS_VERSION}

HEALTHCHECK --interval=15s --timeout=5s --retries=2 --start-period=15s CMD /opt/healthcheck.sh
#HEALTHCHECK --interval=2m --timeout=5s --retries=2 --start-period=2m CMD timeout 3s /usr/local/bin/mc-monitor status-bedrock --host 127.0.0.1 --port $SERVER_PORT || exit 1