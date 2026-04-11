#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH=. ./.venv/bin/python src/train.py --config configs/default.yaml --ablations configs/ablations.yaml --mode full
PYTHONPATH=. ./.venv/bin/python src/evaluate.py --config configs/default.yaml --ablations configs/ablations.yaml --mode full
PYTHONPATH=. ./.venv/bin/python src/analysis.py --config configs/default.yaml
bash scripts/compile_paper.sh
