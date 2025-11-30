ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:3.22
FROM $BUILD_FROM

RUN apk add --no-cache python3 py3-pip

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app /app

# s6-services
COPY rootfs/ /

RUN chmod +x /etc/services.d/api/run \
    && chmod +x /etc/services.d/api/finish

EXPOSE 8099
