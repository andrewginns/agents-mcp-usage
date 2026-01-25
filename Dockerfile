# Docker image for running Merbench evaluations in an isolated environment.
#
# This uses the Playwright Python base image so Chromium dependencies are
# already present for Mermaid CLI / Puppeteer rendering.
FROM mcr.microsoft.com/playwright/python:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install build tools and a modern Node.js (required by mermaid-cli),
# plus uv for dependency management.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libatomic1 \
        curl \
        ca-certificates \
        gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && node --version \
    && npm --version \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# The Playwright base image currently ships with Python 3.10, but this repo
# requires Python 3.12+. Install Python 3.13 via uv and use it via a venv.
RUN uv python install 3.13
RUN uv venv /opt/venv --python 3.13
ENV PATH="/opt/venv/bin:$PATH"

# Install only the dependencies required for Merbench to run inside Docker.
# We intentionally avoid the full project dependency set (which includes
# heavy optional stacks like pandas/streamlit/langchain) to keep builds
# cross-platform and reliable.
COPY agents_mcp_usage/evaluations/mermaid_evals/requirements-merbench.txt ./requirements-merbench.txt
RUN uv pip install --python /opt/venv/bin/python -r requirements-merbench.txt \
    && rm -f requirements-merbench.txt

# Pre-install the pinned mermaid-cli so npx/mmdc do not need to fetch it later.
RUN npm install -g @mermaid-js/mermaid-cli@11.4.2

# Copy the Docker benchmark runner outside the bind mount so it remains
# available even when /workspace is mounted from the host.
COPY agents_mcp_usage/evaluations/mermaid_evals/docker_benchmark.py /usr/local/bin/merbench-runner.py
RUN chmod +x /usr/local/bin/merbench-runner.py

ENTRYPOINT ["python", "/usr/local/bin/merbench-runner.py"]
