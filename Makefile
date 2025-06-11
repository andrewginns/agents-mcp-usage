install:
	uv sync
	npm install -g @mermaid-js/mermaid-cli

lint:
	uv run ruff check .

leaderboard:
	uv run -- streamlit run agents_mcp_usage/multi_mcp/eval_multi_mcp/merbench_ui.py