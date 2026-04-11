#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cd paper
tectonic --only-cached \
  -Z search-path=texmf/lm/tex/latex/lm \
  -Z search-path=texmf/lm/fonts/tfm/public/lm \
  -Z search-path=texmf/lm/fonts/type1/public/lm \
  -Z search-path=texmf/lm/fonts/opentype/public/lm \
  -Z search-path=texmf/lm/fonts/enc/dvips/lm \
  -Z search-path=texmf/lm/fonts/map/dvipdfm/lm \
  -Z search-path=texmf/lm/fonts/map/dvips/lm \
  final.tex
