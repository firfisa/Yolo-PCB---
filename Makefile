# Makefile for PCB Defect Detection System

.PHONY: install test lint format clean docs help

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies and package"
	@echo "  test        - Run all tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code with black"
	@echo "  type-check  - Run type checking with mypy"
	@echo "  clean       - Clean up generated files"
	@echo "  docs        - Generate documentation"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=pcb_detection --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

# Code quality
lint:
	flake8 pcb_detection/ tests/

format:
	black pcb_detection/ tests/

format-check:
	black --check pcb_detection/ tests/

type-check:
	mypy pcb_detection/

# Quality check all
check-all: format-check lint type-check test

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

# Development setup
setup-dev: install-dev
	pre-commit install

# Build package
build:
	python setup.py sdist bdist_wheel

# Run specific test modules
test-core:
	pytest tests/test_core/ -v

test-utils:
	pytest tests/test_utils/ -v