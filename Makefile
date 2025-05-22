install:
	uv sync
	npm install -g @mermaid-js/mermaid-cli

lint:
	uv run ruff check .