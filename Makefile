PYTHON ?= .venv/bin/python

.PHONY: check lint test test-full benchmark figures train

check: lint test

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest -q

# Also runs the optimal 3x3x3 solver with the full pattern databases (about 175 MB, built in about a minute).
test-full:
	RUBIKS_FULL_TESTS=1 $(PYTHON) -m pytest -q

benchmark:
	$(PYTHON) -m rubiks_cube_solvers.benchmark --workers 3

figures:
	$(PYTHON) scripts/make_figures.py
	$(PYTHON) scripts/make_race_gif.py
	$(PYTHON) scripts/export_viewer_data.py

train:
	$(PYTHON) scripts/train_cost_to_go.py --minutes 25 --threads 3
