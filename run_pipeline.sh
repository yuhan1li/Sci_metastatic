#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HERE/config.env" ]]; then source "$HERE/config.env"; else source "$HERE/config.env.example"; fi
MODE="${1:-all}"
cd "$HERE/scripts"

case "$MODE" in
  train)
    python train_full_balanced_split_v4b.py
    ;;
  evaluate)
    python evaluate_heldout_samples_v4b.py
    ;;
  genes)
    python explain_full_v4b.py
    python validate_gene_stability_v4b.py
    python refine_gene_candidates_v4b.py
    python integrate_gene_evidence_v4b.py
    ;;
  all)
    python train_full_balanced_split_v4b.py
    python evaluate_heldout_samples_v4b.py
    python explain_full_v4b.py
    python validate_gene_stability_v4b.py
    python refine_gene_candidates_v4b.py
    python integrate_gene_evidence_v4b.py
    ;;
  *)
    echo "Usage: bash run_pipeline.sh {train|evaluate|genes|all}" >&2; exit 2
    ;;
esac

