PYTHON := ./.venv/bin/python
DEFAULT_CONFIG := configs/default.yaml
ABLATIONS_CONFIG := configs/ablations.yaml

.PHONY: smoke train eval analyze paper all

smoke:
	PYTHONPATH=. $(PYTHON) src/envs.py --sanity
	PYTHONPATH=. $(PYTHON) src/train.py --config $(DEFAULT_CONFIG) --ablations $(ABLATIONS_CONFIG) --mode smoke
	PYTHONPATH=. $(PYTHON) src/evaluate.py --config $(DEFAULT_CONFIG) --ablations $(ABLATIONS_CONFIG) --mode smoke
	PYTHONPATH=. $(PYTHON) src/analysis.py --config $(DEFAULT_CONFIG)

train:
	PYTHONPATH=. $(PYTHON) src/train.py --config $(DEFAULT_CONFIG) --ablations $(ABLATIONS_CONFIG) --mode full

eval:
	PYTHONPATH=. $(PYTHON) src/evaluate.py --config $(DEFAULT_CONFIG) --ablations $(ABLATIONS_CONFIG) --mode full

analyze:
	PYTHONPATH=. $(PYTHON) src/analysis.py --config $(DEFAULT_CONFIG)

paper:
	bash scripts/compile_paper.sh

all: train eval analyze paper
