sources = src/ scripts/ train.py

format:
	uv run ruff format $(sources)

lint:
	uv run ruff check $(sources) --fix --unsafe-fixes

activate:
	source .venv/bin/activate