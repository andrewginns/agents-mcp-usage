# Repository Guidelines

## Project Structure & Module Organization
- `agents_mcp_usage/` is the main Python package.
  - `basic_mcp/` contains **single-server** MCP demos across frameworks (`basic_mcp_use/`).
  - `multi_mcp/` contains **multi-server** MCP demos across frameworks (`multi_mcp_use/`).
  - `factory/` contains model/provider factory helpers (see `MODEL_FACTORY.md` and `model_factory.py`).
  - `evaluations/mermaid_evals/` holds the Merbench benchmark: runners (`evals_pydantic_mcp.py`, `run_multi_evals.py`), Streamlit UI (`merbench_ui.py`), scripts (`scripts/`), and config (`costs.json`, schemas).
- `mcp_servers/` contains local MCP servers used by the agents (e.g., `example_server.py`, `mermaid_validator.py`).
- `docs/` holds images and documentation assets.
- Evaluation CSV/log outputs are written under `mermaid_eval_results/<timestamp>/`.
- Processed/merged dashboard datasets live under `agents_mcp_usage/evaluations/mermaid_evals/results/` (JSON).

## Build, Test, and Development Commands
- `make install` — install Python deps via `uv sync` and globally install Mermaid CLI via `npm` (needed for validation).
- `make upgrade` — refresh deps (`uv sync -U`).
- `make lint` — run `ruff` over the repo.
- `make leaderboard` — launch Merbench Streamlit dashboard.
- `make adk_basic_ui` / `make adk_multi_ui` — start Google ADK web UI for demos.
- Example runs:
  - `uv run agents_mcp_usage/basic_mcp/basic_mcp_use/pydantic_mcp.py`
  - `uv run agents_mcp_usage/basic_mcp/basic_mcp_use/oai-agent_mcp.py`
  - `uv run agents_mcp_usage/basic_mcp/basic_mcp_use/pydantic_mcp_factory.py --list-models`
  - `uv run agents_mcp_usage/evaluations/mermaid_evals/run_multi_evals.py --runs 5 --parallel`

## Coding Style & Naming Conventions
- Python 3.13+; 4‑space indentation; prefer type hints throughout.
- `snake_case` for functions/variables, `UpperCamelCase` for classes, constants in `UPPER_SNAKE_CASE`.
- `ruff` is the linter of record; keep imports clean, avoid unused vars, and fix lint before PRs.

## Testing Guidelines
- No dedicated unit test suite yet. Validate changes by:
  - running the affected demo/eval script with `uv run …`, and/or
  - opening the dashboard on sample CSVs.
- If you add reusable logic, consider introducing lightweight tests in a new `tests/` folder and wire them into the Makefile.

## Commit & Pull Request Guidelines
- Use Conventional Commit prefixes seen in history: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:` (or `refact:`). Keep subjects short and imperative.
- PRs should include: what changed, why, how to run/verify, and any schema/output impacts.
- For UI or JSON schema changes, attach screenshots and/or a small example CSV/JSON.
- Never commit API keys; use `.env` locally and update `.env.example` when adding new config.
