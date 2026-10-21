PYTHON ?= python
WORKERS ?= 4

.PHONY: install test lint smoke main ablations sensitivity experiments report paper all clean

install:  ## editable install with dev tools (CPU torch is enough)
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

smoke:  ## ~20 s end-to-end run of the whole pipeline
	$(PYTHON) -m ultrarl.run --config configs/main.yaml --out results/smoke --smoke --workers $(WORKERS)

main:
	$(PYTHON) -m ultrarl.run --config configs/main.yaml --out results/main --workers $(WORKERS)

ablations:
	$(PYTHON) -m ultrarl.run --config configs/ablations.yaml --out results/ablations --workers $(WORKERS)

sensitivity:
	$(PYTHON) -m ultrarl.run --config configs/sensitivity.yaml --out results/sensitivity --workers $(WORKERS)

experiments: main ablations sensitivity

report:  ## figures + tables from results/ (no training needed)
	$(PYTHON) -m ultrarl.report --results results --paper paper

paper: report
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

all: experiments paper

clean:
	rm -rf results/smoke
	cd paper && latexmk -C main.tex
