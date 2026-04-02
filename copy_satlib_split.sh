#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./copy_satlib_split.sh SOURCE_DIR DEST_DIR [DATASET_DIR ...]

Copy legacy SATLIB CNF files into train/valid/test split directories using the
same split boundaries as dataset.py:
  test:  1-100
  valid: 101-200
  train: 201-1000

Special case:
  uf250-* and uuf250-* only ship with the test split, so only test files are
  copied for those datasets.

Arguments:
  SOURCE_DIR   Directory containing legacy SATLIB folders such as uf50-218.
  DEST_DIR     Output directory where each dataset folder will be created with
               train/, valid/, and test/ subdirectories.
  DATASET_DIR  Optional dataset folder names to copy. If omitted, all
               immediate child directories matching uf* or uuf* are copied.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 1
fi

source_root=$1
dest_root=$2
shift 2

if [[ ! -d "$source_root" ]]; then
    echo "Source directory not found: $source_root" >&2
    exit 1
fi

declare -a datasets=()
if [[ $# -gt 0 ]]; then
    datasets=("$@")
else
    shopt -s nullglob
    for path in "$source_root"/uf* "$source_root"/uuf*; do
        [[ -d "$path" ]] || continue
        datasets+=("$(basename "$path")")
    done
    shopt -u nullglob
fi

if [[ ${#datasets[@]} -eq 0 ]]; then
    echo "No SATLIB dataset directories found under $source_root" >&2
    exit 1
fi

copy_split() {
    local dataset_dir=$1
    local split_name=$2
    local start_idx=$3
    local end_idx=$4
    local prefix=$5
    local src_dir=$6
    local dst_dir=$7

    mkdir -p "$dst_dir"

    local idx file_name src_file dst_file split_idx
    split_idx=0
    for ((idx = start_idx; idx <= end_idx; idx++)); do
        file_name="${prefix}-0${idx}.cnf"
        src_file="$src_dir/$file_name"
        if [[ ! -f "$src_file" ]]; then
            echo "Missing source file for $dataset_dir ($split_name): $src_file" >&2
            exit 1
        fi
        printf -v dst_file "%s/%05d.cnf" "$dst_dir" "$split_idx"
        sanitize_cnf "$src_file" "$dst_file"
        ((split_idx += 1))
    done
}

sanitize_cnf() {
    local src_file=$1
    local dst_file=$2

    # Strip only terminal lines that are empty or exactly "%" or "0"; keep any
    # such lines if more CNF content follows later in the file.
    awk '
        /^[[:space:]]*$/ || /^[[:space:]]*(%|0)[[:space:]]*$/ {
            trailing[++trailing_count] = $0
            next
        }
        {
            if (trailing_count > 0) {
                for (i = 1; i <= trailing_count; i++) {
                    print trailing[i]
                }
                trailing_count = 0
                delete trailing
            }
            print
        }
    ' "$src_file" > "$dst_file"
}

is_test_only_dataset() {
    local dataset_dir=$1
    [[ "$dataset_dir" == uf250-* || "$dataset_dir" == uuf250-* ]]
}

for dataset_dir in "${datasets[@]}"; do
    src_dir="$source_root/$dataset_dir"
    if [[ ! -d "$src_dir" ]]; then
        echo "Dataset directory not found: $src_dir" >&2
        exit 1
    fi

    prefix=${dataset_dir%%-*}
    if [[ -z "$prefix" || "$prefix" == "$dataset_dir" ]]; then
        echo "Could not infer CNF filename prefix from dataset directory: $dataset_dir" >&2
        exit 1
    fi

    if ! is_test_only_dataset "$dataset_dir"; then
        copy_split "$dataset_dir" "train" 201 1000 "$prefix" "$src_dir" "$dest_root/$dataset_dir/train"
        copy_split "$dataset_dir" "valid" 101 200 "$prefix" "$src_dir" "$dest_root/$dataset_dir/valid"
    fi
    copy_split "$dataset_dir" "test" 1 100 "$prefix" "$src_dir" "$dest_root/$dataset_dir/test"
done

echo "Copied ${#datasets[@]} SATLIB dataset(s) into $dest_root"
