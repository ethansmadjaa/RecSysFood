.PHONY: front back build help

help:
	@echo "Available commands:"
	@echo "  make front  - Install frontend dependencies and start dev server"
	@echo "  make back   - Install uv, sync dependencies, and run FastAPI server"
	@echo "  make build  - Build frontend for production"

front:
	cd frontend && pnpm i && pnpm dev

back:
	pip install uv && uv sync && uv run main.py

build:
	cd frontend && pnpm i && pnpm build
