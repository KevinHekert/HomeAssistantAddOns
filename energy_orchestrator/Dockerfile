ARG BUILD_FROM
FROM $BUILD_FROM AS builder

# Build dependencies for compiling packages like scikit-learn
RUN apk add --no-cache python3 py3-pip python3-dev \
    gcc g++ gfortran musl-dev openblas-dev

WORKDIR /app

# Python dependencies - install to a specific location
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages --target=/install -r requirements.txt

# Final stage
FROM $BUILD_FROM

# Runtime dependencies
RUN apk add --no-cache python3 py3-pip openblas libgfortran libstdc++

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local/lib/python-packages

# Set PYTHONPATH to include the installed packages
ENV PYTHONPATH="/usr/local/lib/python-packages:${PYTHONPATH}"

WORKDIR /app

# Jouw Flask-code
COPY app /app

# Entrypoint script
COPY run.sh /
RUN chmod +x /run.sh

EXPOSE 8099

CMD ["/run.sh"]
