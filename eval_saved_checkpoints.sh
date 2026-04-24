#!/bin/bash
DATASETS=(
  "50"
  "3-sat/v5c24"
  "3-sat/v10c49"
  "3-sat/v15c71"
  "3-sat/v20c92"
  "sr/easy"
  "3-sat/easy"
  "ca/easy"
  "ps/easy"
  "k-clique/easy"
  "k-domset/easy"
  "k-vercov/easy"
)

for DATASET in "${DATASETS[@]}"; do
  if [[ "${DATASET}" == "50" ]]; then
    DATASET_SPEC="50"
    CHECKPOINT_TAG="sat50"
  else
    DATASET_SPEC="${DATASET}"
    CHECKPOINT_TAG="${DATASET//\//_}"
  fi

  for SEED in 0 1 2 3 4; do
    CHECKPOINT_PATH="checkpoints/graphqsat/${CHECKPOINT_TAG}_seed${SEED}.pth"

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
