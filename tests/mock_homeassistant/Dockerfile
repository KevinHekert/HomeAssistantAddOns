FROM python:3.12-slim

WORKDIR /app

# Install Flask
RUN pip install --no-cache-dir Flask==3.1.2

# Copy mock API
COPY app.py .

EXPOSE 8123

CMD ["python", "app.py"]
