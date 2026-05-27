.PHONY: install lint format test build

install:
	uv sync

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest tests/ -q

build:
	uv run llm build run
