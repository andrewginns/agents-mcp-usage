install:
	uv sync
	npm install -g @mermaid-js/mermaid-cli

upgrade:
	uv sync -U

lint:
	uv run ruff check .

leaderboard:
	uv run -- streamlit run agents_mcp_usage/evaluations/mermaid_evals/merbench_ui.py

adk_basic_ui:
	uv run adk web agents_mcp_usage/basic_mcp

adk_multi_ui:
	uv run adk web agents_mcp_usage/multi_mcp

merbench-docker-build:
	docker build -t merbench .

merbench-docker-run:
	docker run --rm \
	  -e GEMINI_API_KEY=$${GEMINI_API_KEY} \
	  -e OPENAI_API_KEY=$${OPENAI_API_KEY} \
	  -v "$$(pwd)/mermaid_eval_results:/app/mermaid_eval_results" \
	  merbench $${ARGS}
