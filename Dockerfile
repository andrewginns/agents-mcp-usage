# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install Node.js, npm, and Chromium dependencies for mermaid-cli
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-liberation \
        fonts-noto-color-emoji \
        libasound2 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libatspi2.0-0 \
        libcairo2 \
        libcups2 \
        libdrm2 \
        libgbm1 \
        libgtk-3-0 \
        libnss3 \
        libpango-1.0-0 \
        libx11-xcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxi6 \
        libxrandr2 \
        libxrender1 \
        libxshmfence1 \
        libxss1 \
        libxtst6 \
        nodejs \
        npm \
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
