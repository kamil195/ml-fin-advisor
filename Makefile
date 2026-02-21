.PHONY: install install-dev lint test serve clean docker-build docker-up docker-down docker-logs docker-test

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-all:
	pytest tests/ -v --cov=src --cov-report=term-missing

serve:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

generate-data:
	python -m src.data.mock_generator

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache dist build

# ── Docker ──────────────────────────────────────────────────────────

docker-build:
	docker compose build --no-cache

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs --tail=100

docker-test:
	python test_serving.py --base-url http://localhost:8000 --no-wait

docker-restart: docker-down docker-build docker-up
