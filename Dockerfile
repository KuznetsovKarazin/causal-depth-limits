FROM python:3.12-slim
WORKDIR /app
# python:3.12-slim ships without make/gcc; install build essentials for `make all`
RUN apt-get update && apt-get install -y --no-install-recommends \
        make build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# fallback if `make` is unavailable for any reason: run the pipeline directly
CMD ["make", "all"]
