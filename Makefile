PY := .venv/bin/python

.PHONY: check test typecheck audit
check:
	$(PY) -m hidrometros.check

test:
	$(PY) -m pytest -q

typecheck:
	$(PY) -m mypy

audit:
	$(PY) -m hidrometros.check audit
