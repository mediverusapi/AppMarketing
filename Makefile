PYTHON?=.venv/bin/python
PIP?=.venv/bin/pip

.PHONY: help venv install run test clean web api dev

help:
	@echo "Targets:"
	@echo "  venv     - create virtual environment"
	@echo "  install  - install Python dependencies"
	@echo "  run      - run pipeline (VIDEO=<path> [ARGS=...])"
	@echo "  test     - run tests"
	@echo "  clean    - remove intermediate files in data/working"

venv:
	python3 -m venv .venv
	. .venv/bin/activate; $(PIP) install -U pip wheel setuptools

install: venv
	$(PIP) install -r requirements.txt
	@echo "Note: Install PyTorch separately if needed (see README)."

run:
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make run VIDEO=data/input/sample.mp4 [ARGS=...]"; exit 1; fi
	$(PYTHON) -m src.main $(VIDEO) $(ARGS)

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf data/working/*

api:
	$(PYTHON) -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload

web:
	cd web && npm install && npm run dev

dev:
	@echo "Start API: make api" 
	@echo "Start Web: make web"


