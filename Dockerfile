# Multi-stage Dockerfile for kvraft nodes.
# Skeleton only — Day 1 evening will refine once the application code exists.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dep installs
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project metadata first so dep layer is cached across code edits
COPY pyproject.toml ./

RUN uv pip install --system -e .

# Copy source last — changes here don't invalidate dep cache
COPY src ./src
COPY scripts ./scripts

EXPOSE 8000 4321

# Entrypoint stubbed until API module exists (Day 1 morning)
CMD ["python", "-c", "print('kvraft container — application entrypoint pending Day 1')"]
