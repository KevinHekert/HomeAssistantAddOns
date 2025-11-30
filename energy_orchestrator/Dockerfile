ARG BUILD_FROM
FROM $BUILD_FROM

# Python + pip
RUN apk add --no-cache python3 py3-pip

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Django project code
COPY app /app

# Entrypoint
COPY run.sh /
RUN chmod +x /run.sh

EXPOSE 8099

CMD ["/run.sh"]
