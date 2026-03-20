# GAIO.py — developer convenience targets
# Usage: make <target>
# Requires: conda env `gaio` to be active

.PHONY: install install-gpu install-mpi install-all \
        test test-cov test-fast \
        clean clean-all \
        check

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[dev,gpu]"

install-mpi:
	pip install -e ".[dev,mpi]"

install-all:
	pip install -e ".[dev,gpu,mpi]"

# ── Testing ───────────────────────────────────────────────────────────────────

# Run the full test suite with verbose output
test:
	pytest tests/ -v

# Run tests and show per-line coverage for the gaio package
test-cov:
	pytest tests/ -v --cov=gaio --cov-report=term-missing

# Run only the fast unit tests (skip anything marked slow)
test-fast:
	pytest tests/ -v -m "not slow"

# Run a specific test file, e.g.: make test-file FILE=tests/core/test_box.py
test-file:
	pytest $(FILE) -v

# ── Sanity check ──────────────────────────────────────────────────────────────

# Quick import smoke test — useful after environment changes
check:
	python -c "from gaio import Box, BoxPartition, BoxSet; print('gaio import OK')"

# ── Cleanup ───────────────────────────────────────────────────────────────────

# Remove generated files that should not be committed
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "numba_cache" -exec rm -rf {} + 2>/dev/null || true

# Also remove build artefacts (dist/, build/)
clean-all: clean
	rm -rf dist/ build/
