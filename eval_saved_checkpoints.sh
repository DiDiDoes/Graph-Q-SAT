#!/bin/bash
DATASETS=(
  "50"
  "sr"
  "3-sat"
  "ca"
  "ps"
  "k-clique"
  "k-domset"
  "k-vercov"
)

for DATASET in "${DATASETS[@]}"; do
  if [[ "${DATASET}" == "50" ]]; then
    DATASET_SPEC="50"
    CHECKPOINT_TAG="sat50"
  else
    DATASET_SPEC="${DATASET}/easy"
    CHECKPOINT_TAG="${DATASET}_easy"
  fi

  for SEED in 0 1 2 3 4; do
    CHECKPOINT_PATH="checkpoints/graphqsat_${CHECKPOINT_TAG}_seed${SEED}.pth"

    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
      echo "Missing checkpoint: ${CHECKPOINT_PATH}" >&2
      exit 1
    fi

    echo "=== Evaluating dataset ${DATASET} with seed ${SEED} ==="
    python main.py test \
      --dataset "${DATASET_SPEC}" \
      --sat true \
      --seed "${SEED}" \
      --checkpoint-path "${CHECKPOINT_PATH}"
    echo
  done
done
