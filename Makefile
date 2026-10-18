PYTHON ?= .venv/bin/python

.PHONY: check lint test

check: lint test

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest -q
