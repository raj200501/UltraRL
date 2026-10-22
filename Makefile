PYTHON ?= python
WORKERS ?= 4
EXPERIMENTS := main ablations sensitivity followups encoders process

.PHONY: install test lint smoke analyses report paper experiments reproduce equivalence clean $(EXPERIMENTS)

install:  ## editable install with dev tools (CPU torch is enough)
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

smoke:  ## ~30 s end-to-end run of the whole pipeline in a scratch directory
	rm -rf results/smoke
	$(PYTHON) -m ultrarl.run --config configs/main.yaml --out results/smoke --smoke --workers $(WORKERS)

analyses:  ## observation coverage (hand-coded and agents' own rollouts) and the value probe, from saved checkpoints (~20 min)
	$(PYTHON) scripts/observation_coverage.py
	$(PYTHON) scripts/agent_coverage.py
	$(PYTHON) scripts/q_probe.py

report:  ## every figure, table and number in the paper, from the saved results (no training)
	$(PYTHON) -m ultrarl.report --results results --paper paper

paper: report  ## main.tex omits embedded dates and IDs, so the PDF depends only on its sources
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# Resume or extend the saved experiments in results/: runs whose checkpoints
# already exist (and match the config) are skipped, evaluation is recomputed.
$(EXPERIMENTS):
	$(PYTHON) -m ultrarl.run --config configs/$@.yaml --out results/$@ --workers $(WORKERS)

experiments: $(EXPERIMENTS)

# Retrain everything from scratch into results_repro/ (~8 h on 4 cores) and
# check that every checkpoint is bitwise identical to the saved one (same package
# versions and an AVX-512 CPU; elsewhere use scripts/compare_summaries.py).
reproduce:
	@for exp in $(EXPERIMENTS); do \
	  $(PYTHON) -m ultrarl.run --config configs/$$exp.yaml --out results_repro/$$exp --workers $(WORKERS) || exit 1; \
	  $(PYTHON) scripts/compare_checkpoints.py results/$$exp results_repro/$$exp || exit 1; \
	done

equivalence:  ## rewritten environment vs. the original implementation, step by step
	$(PYTHON) scripts/check_legacy_equivalence.py

clean:
	rm -rf results/smoke results_repro
	cd paper && latexmk -c main.tex
