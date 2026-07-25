#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

module load miniconda3
set +u
source ~/.bashrc
set -u
conda activate "${CONDA_ENV:-hippynn-expanded-lmax}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/benchmarks/run_models/.matplotlib-cache}"
export HIPPYNN_USE_CUSTOM_KERNELS="${HIPPYNN_USE_CUSTOM_KERNELS:-True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HIPPYNN_SOURCE_ROOT="${HIPPYNN_SOURCE_ROOT:-/vast/home/logan_bolton/Github/hippynn-optimizations-expanded}"
HIPPYNN_INVARIANTS_REF="${HIPPYNN_INVARIANTS_REF:-lmax_3_nmax_5}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/rotating_ring/results/z_rotation_radius1_no_gap_l3_n5_all_sizes}"
RESULTS_JSON_DIR="${RESULTS_JSON_DIR:-$OUTPUT_DIR/json_results}"
EPOCHS="${EPOCHS:-500}"
SEEDS=(${SEEDS:-0 1})
DEVICES=(${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7})
PARALLEL_CONFIGS="${PARALLEL_CONFIGS:-${#DEVICES[@]}}"
RING_GRAPH_CONFIGS=(${RING_GRAPH_CONFIGS:- \
    2d:1:1 2d:1:2 2d:1:3 2d:1:4 \
    2d:2:1 2d:2:2 2d:2:3 2d:2:4 \
    2d:3:1 2d:3:2 2d:3:3 2d:3:4 \
    2d:4:1 2d:4:2 2d:4:3 2d:4:4 \
})

if [[ ! -d "$HIPPYNN_SOURCE_ROOT/hippynn" ]]; then
    echo "Error: no hippynn package under HIPPYNN_SOURCE_ROOT=$HIPPYNN_SOURCE_ROOT" >&2
    exit 1
fi

# The HIP-HOP invariant implementation and its interaction wiring must exactly
# match the requested branch, even if HIPPYNN_SOURCE_ROOT is on another branch.
INVARIANT_FILES=(
    hippynn/layers/hiplayers/invariants.py
    hippynn/layers/hiplayers/interactions.py
    hippynn/networks/hiphop.py
)
git -C "$HIPPYNN_SOURCE_ROOT" rev-parse --verify "$HIPPYNN_INVARIANTS_REF^{commit}" >/dev/null
if ! git -C "$HIPPYNN_SOURCE_ROOT" diff --quiet "$HIPPYNN_INVARIANTS_REF" -- "${INVARIANT_FILES[@]}"; then
    echo "Error: HIPPYNN invariant code does not match ref $HIPPYNN_INVARIANTS_REF." >&2
    echo "Use a clean checkout/worktree of that branch via HIPPYNN_SOURCE_ROOT." >&2
    exit 1
fi

export PYTHONPATH="$HIPPYNN_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
IMPORTED_HIPPYNN="$($PYTHON_BIN -c 'import pathlib, hippynn; print(pathlib.Path(hippynn.__file__).resolve().parents[1])')"
if [[ "$IMPORTED_HIPPYNN" != "$(realpath "$HIPPYNN_SOURCE_ROOT")" ]]; then
    echo "Error: Python imported hippynn from $IMPORTED_HIPPYNN, expected $HIPPYNN_SOURCE_ROOT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "HIPPYNN source: $IMPORTED_HIPPYNN"
echo "Invariant ref: $HIPPYNN_INVARIANTS_REF ($(git -C "$HIPPYNN_SOURCE_ROOT" rev-parse --short "$HIPPYNN_INVARIANTS_REF"))"
echo "Model: HIP-HOP l_max=3 n_max=5"
echo "Dataset: z-phase radius=1, gap=0, far inner rotation=15 deg, all 1..4 inner/outer sizes"
nvidia-smi -L

"$PYTHON_BIN" benchmarks/run_models/sweep.py \
    --dataset rotating_ring \
    --ring-z-phase-sample \
    --ring-n-graphs 2 \
    --ring-inner-radius 1 \
    --ring-outer-gap 0 \
    --ring-z-phase-far-inner-rotation-deg 15 \
    --ring-graph-configs "${RING_GRAPH_CONFIGS[@]}" \
    --epochs "$EPOCHS" \
    --seeds "${SEEDS[@]}" \
    --hard-cutoffs 7 \
    --interaction-layers 1 \
    --model-configs l3_n5 \
    --model hiphop \
    --neighborhood-cutoff edges \
    --output-dir "$OUTPUT_DIR" \
    --results-json-dir "$RESULTS_JSON_DIR" \
    --parallel-configs "$PARALLEL_CONFIGS" \
    --devices "${DEVICES[@]}"
