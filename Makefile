.PHONY: lint test ci ci-docker linear experiment analysis figures

lint:
	uv run black --check .
	uv run isort --check-only .
	uv run flake8 .

test:
	uv run pytest -q

ci: lint test

ci-docker:
	docker run --rm -v "$(CURDIR)":/src -w /src python:3.12-slim \
		sh -c "pip install -q uv && export UV_PROJECT_ENVIRONMENT=/tmp/venv && \
		uv run black --check . && uv run isort --check-only . && \
		uv run flake8 . && uv run pytest -q"

linear:
	uv run python linear_sim.py

experiment:
	uv run python experiment.py

analysis:
	uv run python analysis.py

figures: analysis
	uv run python figures.py
