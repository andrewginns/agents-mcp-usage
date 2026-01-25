MERBENCH_IMAGE ?= merbench:latest
RUNS ?= 15
JUDGE_MODEL ?= gemini-2.5-pro
TIMEOUT ?= 600
OUTPUT_DIR ?= ./mermaid_eval_results
PARALLEL ?= 0
SEQUENTIAL ?= 0
DEBUG_TRACES ?= 0

ENV_FILE_FLAG := $(if $(wildcard .env),--env-file .env,)

# Pass through common provider credentials if they exist in the host env.
COMMON_ENV_VARS := \
	-e OPENAI_API_KEY \
	-e GEMINI_API_KEY \
	-e GOOGLE_API_KEY \
	-e ANTHROPIC_API_KEY \
	-e AWS_REGION \
	-e AWS_PROFILE \
	-e PERPLEXITY_API_KEY \
	-e OPENROUTER_API_KEY \
	-e OPENROUTER_APP_URL \
	-e OPENROUTER_APP_TITLE \
	-e LOGFIRE_TOKEN \
	-e LOCAL_OPENAI_BASE_URL \
	-e OLLAMA_BASE_URL

.PHONY: install upgrade lint leaderboard adk_basic_ui adk_multi_ui benchmark-image benchmark benchmark-dry-run

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

benchmark-image:
	docker build -t $(MERBENCH_IMAGE) .

benchmark: benchmark-image
	@if [ -z "$(MODEL)" ]; then \
		echo "MODEL is required. Example: make benchmark MODEL='openai:gpt-5.1 (none)' RUNS=5"; \
		exit 1; \
	fi
	docker run --rm \
		$(ENV_FILE_FLAG) \
		$(COMMON_ENV_VARS) \
		-e MODEL="$(MODEL)" \
		-e RUNS="$(RUNS)" \
		-e JUDGE_MODEL="$(JUDGE_MODEL)" \
		-e TIMEOUT="$(TIMEOUT)" \
		-e OUTPUT_DIR="$(OUTPUT_DIR)" \
		-e PARALLEL="$(PARALLEL)" \
		-e SEQUENTIAL="$(SEQUENTIAL)" \
		-e DEBUG_TRACES="$(DEBUG_TRACES)" \
		-e TRACE_DIR="$(TRACE_DIR)" \
		-e BENCHMARK_FLAGS="$(BENCHMARK_FLAGS)" \
		-v "$(PWD)":/workspace \
		-w /workspace \
		$(MERBENCH_IMAGE)

benchmark-dry-run: benchmark-image
	@if [ -z "$(MODEL)" ]; then \
		echo "MODEL is required. Example: make benchmark-dry-run MODEL='openai:gpt-5.1 (none)'"; \
		exit 1; \
	fi
	docker run --rm \
		$(ENV_FILE_FLAG) \
		$(COMMON_ENV_VARS) \
		-e DRY_RUN="1" \
		-e MODEL="$(MODEL)" \
		-e RUNS="$(RUNS)" \
		-e JUDGE_MODEL="$(JUDGE_MODEL)" \
		-e TIMEOUT="$(TIMEOUT)" \
		-e OUTPUT_DIR="$(OUTPUT_DIR)" \
		-e PARALLEL="$(PARALLEL)" \
		-e SEQUENTIAL="$(SEQUENTIAL)" \
		-e DEBUG_TRACES="$(DEBUG_TRACES)" \
		-e TRACE_DIR="$(TRACE_DIR)" \
		-e BENCHMARK_FLAGS="$(BENCHMARK_FLAGS)" \
		-v "$(PWD)":/workspace \
		-w /workspace \
		$(MERBENCH_IMAGE)
