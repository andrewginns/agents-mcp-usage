# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install Node.js and npm for mermaid-cli
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install mermaid CLI globally
RUN npm install -g @mermaid-js/mermaid-cli

# Set work directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Copy source
COPY . .

# Sync dependencies during build so they are baked into the image
RUN uv sync --frozen

# Default entrypoint
ENTRYPOINT ["/app/docker/merbench-entrypoint.sh"]
