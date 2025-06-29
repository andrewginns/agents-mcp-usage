install:
	uv sync
	npm install -g @mermaid-js/mermaid-cli

lint:
	uv run ruff check .

leaderboard:
	uv run -- streamlit run agents_mcp_usage/evaluations/mermaid_evals/merbench_ui.py

adk_basic_ui:
	uv run adk web agents_mcp_usage/basic_mcp

adk_multi_ui:
	uv run adk web agents_mcp_usage/multi_mcp