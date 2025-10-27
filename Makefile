# Makefile for Token Calculator API

.PHONY: help install dev test lint format clean docker-build docker-run docker-stop

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	poetry install

dev: ## Run development server
	poetry run python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	poetry run pytest tests/ -v

test-cov: ## Run tests with coverage
	poetry run pytest tests/ -v --cov=app --cov-report=html --cov-report=term

lint: ## Run linter
	poetry run ruff check app/ tests/

format: ## Format code
	poetry run black app/ tests/
	poetry run ruff check --fix app/ tests/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

docker-build: ## Build Docker image
	docker build -t token-calculator-api .

docker-run: ## Run Docker container
	docker run -d --name token-calculator-api -p 8000:8000 token-calculator-api

docker-stop: ## Stop Docker container
	docker stop token-calculator-api
	docker rm token-calculator-api

docker-compose-up: ## Start with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop docker-compose
	docker-compose down

docker-compose-logs: ## View docker-compose logs
	docker-compose logs -f

check: lint test ## Run linting and tests

all: clean install check ## Clean, install, lint, and test
