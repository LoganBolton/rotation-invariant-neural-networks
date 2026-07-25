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
# Do not let ~/.local packages override the activated conda environment. On the
# cluster, the user-site PyTorch may target a newer CUDA driver than the node.
export PYTHONNOUSERSITE=1
# The benchmark does not require the optional Triton/Numba/CuPy message-passing
# kernels. False allows HIPPyNN to use its PyTorch implementation when absent.
export HIPPYNN_USE_CUSTOM_KERNELS=False
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HIPHOP_SOURCE_ROOT="${HIPHOP_SOURCE_ROOT:-/vast/home/logan_bolton/Github/hippynn-optimizations-expanded}"
HIPHOP_INVARIANTS_REF="${HIPHOP_INVARIANTS_REF:-lmax3-basis-invariants-tests}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/rotating_ring/results/z_rotation_radius1_no_gap_basic_l2_l3_n4_all_sizes}"
RESULTS_JSON_DIR="${RESULTS_JSON_DIR:-$OUTPUT_DIR/json_results}"
EPOCHS="${EPOCHS:-500}"
SEEDS=(${SEEDS:-0 1})
DEVICES=(cuda:0)
PARALLEL_CONFIGS=1
HIPHOP_GROUP_NORM="${HIPHOP_GROUP_NORM:-true}"
RING_Z_PHASE_SAMPLE="${RING_Z_PHASE_SAMPLE:-true}"
RING_GRAPH_CONFIGS=(${RING_GRAPH_CONFIGS:- \
    2d:1:1 2d:1:2 2d:1:3 2d:1:4 \
    2d:2:1 2d:2:2 2d:2:3 2d:2:4 \
    2d:3:1 2d:3:2 2d:3:3 2d:3:4 \
    2d:4:1 2d:4:2 2d:4:3 2d:4:4 \
})

case "${HIPHOP_GROUP_NORM,,}" in
    true|1|yes|on)
        GROUP_NORM_ARGS=(--hiphop-group-norm)
        ;;
    false|0|no|off)
        GROUP_NORM_ARGS=(--no-hiphop-group-norm)
        ;;
    *)
        echo "Error: HIPHOP_GROUP_NORM must be true or false, got $HIPHOP_GROUP_NORM" >&2
        exit 2
        ;;
esac

case "${RING_Z_PHASE_SAMPLE,,}" in
    true|1|yes|on)
        RING_VARIANT_ARGS=(--ring-z-phase-sample --ring-z-phase-far-inner-rotation-deg 15)
        RING_DESCRIPTION="z-phase radius=1, gap=0, far inner rotation=15 deg"
        ;;
    false|0|no|off)
        RING_VARIANT_ARGS=()
        RING_DESCRIPTION="ordinary two-graph phase pair, radius=1, gap=0"
        ;;
    *)
        echo "Error: RING_Z_PHASE_SAMPLE must be true or false, got $RING_Z_PHASE_SAMPLE" >&2
        exit 2
        ;;
esac

if [[ ! -d "$HIPHOP_SOURCE_ROOT/hippynn" ]]; then
    echo "Error: no hippynn package under HIPHOP_SOURCE_ROOT=$HIPHOP_SOURCE_ROOT" >&2
    exit 1
fi

INVARIANT_FILES=(
    hippynn/layers/hiplayers/invariants.py
    hippynn/layers/hiplayers/interactions.py
    hippynn/networks/hiphop.py
)
git -C "$HIPHOP_SOURCE_ROOT" rev-parse --verify "$HIPHOP_INVARIANTS_REF^{commit}" >/dev/null
if ! git -C "$HIPHOP_SOURCE_ROOT" diff --quiet "$HIPHOP_INVARIANTS_REF" -- "${INVARIANT_FILES[@]}"; then
    echo "Error: HIPPyNN invariant code does not match the basic-basis ref $HIPHOP_INVARIANTS_REF." >&2
    echo "Use a clean checkout/worktree of that branch via HIPHOP_SOURCE_ROOT." >&2
    exit 1
fi

export PYTHONPATH="$HIPHOP_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
IMPORTED_HIPPYNN="$($PYTHON_BIN -c 'import pathlib, hippynn; print(pathlib.Path(hippynn.__file__).resolve().parents[1])')"
if [[ "$IMPORTED_HIPPYNN" != "$(realpath "$HIPHOP_SOURCE_ROOT")" ]]; then
    echo "Error: Python imported hippynn from $IMPORTED_HIPPYNN, expected $HIPHOP_SOURCE_ROOT" >&2
    exit 1
fi

if [[ " ${DEVICES[*]} " == *" cuda"* ]]; then
    if [[ "$($PYTHON_BIN -c 'import torch; print(torch.cuda.is_available())')" != "True" ]]; then
        "$PYTHON_BIN" -c 'import sys, torch; print(f"Python: {sys.executable}"); print(f"PyTorch: {torch.__version__} ({torch.__file__})"); print(f"PyTorch CUDA build: {torch.version.cuda}")' >&2
        echo "Error: CUDA is unavailable to the activated conda environment on this node." >&2
        exit 1
    fi
fi

mkdir -p "$OUTPUT_DIR"
echo "HIPPYNN source: $IMPORTED_HIPPYNN"
echo "Basic invariant ref: $HIPHOP_INVARIANTS_REF ($(git -C "$HIPHOP_SOURCE_ROOT" rev-parse --short "$HIPHOP_INVARIANTS_REF"))"
echo "Models: HIP-HOP l_max=3 n_max=4; HIP-HOP l_max=2 n_max=4"
echo "HIP-HOP GroupNorm: $HIPHOP_GROUP_NORM"
echo "Dataset: $RING_DESCRIPTION, all 1..4 inner/outer sizes"
if [[ " ${DEVICES[*]} " == *" cuda"* ]]; then
    nvidia-smi -L
fi

"$PYTHON_BIN" benchmarks/run_models/sweep.py \
    --dataset rotating_ring \
    "${RING_VARIANT_ARGS[@]}" \
    --ring-n-graphs 2 \
    --ring-inner-radius 1 \
    --ring-outer-gap 0 \
    --ring-graph-configs "${RING_GRAPH_CONFIGS[@]}" \
    --epochs "$EPOCHS" \
    --seeds "${SEEDS[@]}" \
    --hard-cutoffs 7 \
    --interaction-layers 1 \
    --model-configs l3_n4 l2_n4 \
    --model hiphop \
    "${GROUP_NORM_ARGS[@]}" \
    --neighborhood-cutoff edges \
    --output-dir "$OUTPUT_DIR" \
    --results-json-dir "$RESULTS_JSON_DIR" \
    --parallel-configs "$PARALLEL_CONFIGS" \
    --devices "${DEVICES[@]}"
