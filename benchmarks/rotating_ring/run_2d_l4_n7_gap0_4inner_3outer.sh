#!/usr/bin/env bash
set -euo pipefail

cd /vast/home/logan_bolton/Github/rotation-invariant-neural-networks

module load miniconda3
set +u
source ~/.bashrc
set -u
conda activate "${CONDA_ENV:-hippynn-expanded-lmax}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/benchmarks/run_models/.matplotlib-cache}"
export HIPPYNN_USE_CUSTOM_KERNELS="${HIPPYNN_USE_CUSTOM_KERNELS:-True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HIPPYNN_SOURCE_ROOT="${HIPPYNN_SOURCE_ROOT:-/vast/home/logan_bolton/Github/hippynn-optimizations-expanded}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/rotating_ring/results/2d_l4_n7_gap0}"
RESULTS_JSON_DIR="${RESULTS_JSON_DIR:-${OUTPUT_DIR}/json_results}"
EPOCHS="${EPOCHS:-500}"
RING_N_GRAPHS="${RING_N_GRAPHS:-2}"
RING_SEED="${RING_SEED:-0}"
RING_INNER_RADIUS="${RING_INNER_RADIUS:-1.0}"
RING_OUTER_GAP="${RING_OUTER_GAP:-0.0}"
SEEDS=(${SEEDS:-0 1})
HARD_CUTOFFS=(${HARD_CUTOFFS:-7.0})
INTERACTION_LAYERS=(${INTERACTION_LAYERS:-1})
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$OUTPUT_DIR"

if [ ! -d "$HIPPYNN_SOURCE_ROOT/hippynn" ]; then
    echo "Error: HIPPYNN_SOURCE_ROOT does not contain a hippynn package: $HIPPYNN_SOURCE_ROOT"
    exit 1
fi

export PYTHONPATH="$HIPPYNN_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RING_OUTER_RADIUS="$("$PYTHON_BIN" -c "print(float('$RING_INNER_RADIUS') + float('$RING_OUTER_GAP'))")"

echo "Host: $(hostname)"
echo "Python: $($PYTHON_BIN --version)"
echo "HIPPYNN_SOURCE_ROOT: $HIPPYNN_SOURCE_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Ring geometry: inner_radius=$RING_INNER_RADIUS outer_gap=$RING_OUTER_GAP outer_radius=$RING_OUTER_RADIUS"
echo "Graph config: 2d_4inner_3_outer"
echo "Device: $DEVICE"
nvidia-smi -L

"$PYTHON_BIN" benchmarks/run_models/sweep.py \
    --dataset rotating_ring \
    --ring-n-graphs "$RING_N_GRAPHS" \
    --ring-seed "$RING_SEED" \
    --ring-inner-radius "$RING_INNER_RADIUS" \
    --ring-outer-gap "$RING_OUTER_GAP" \
    --ring-graph-configs 2d_4inner_3_outer \
    --epochs "$EPOCHS" \
    --seeds "${SEEDS[@]}" \
    --hard-cutoffs "${HARD_CUTOFFS[@]}" \
    --interaction-layers "${INTERACTION_LAYERS[@]}" \
    --model-configs l4_n7 \
    --model hiphop \
    --neighborhood-cutoff edges \
    --output-dir "$OUTPUT_DIR" \
    --results-json-dir "$RESULTS_JSON_DIR" \
    --parallel-configs 1 \
    --device "$DEVICE"
