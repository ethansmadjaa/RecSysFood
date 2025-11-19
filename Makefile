.PHONY: front back help

help:
	@echo "Available commands:"
	@echo "  make front  - Install frontend dependencies and start dev server"
	@echo "  make back   - Install uv, sync dependencies, and run FastAPI server"

front:
	cd frontend && pnpm i && pnpm dev

back:
	pip install uv && uv sync && uv run fastapi dev main.py
