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
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/rotating_ring/results/2d_l2_l3_n2to5_gap0}"
RESULTS_JSON_DIR="${RESULTS_JSON_DIR:-${OUTPUT_DIR}/json_results}"
LOG_DIR="${LOG_DIR:-benchmarks/rotating_ring/results/local_logs}"
EPOCHS="${EPOCHS:-500}"
RING_N_GRAPHS="${RING_N_GRAPHS:-2}"
RING_SEED="${RING_SEED:-0}"
RING_INNER_RADIUS="${RING_INNER_RADIUS:-1.0}"
RING_OUTER_GAP="${RING_OUTER_GAP:-0.0}"
SEEDS=(${SEEDS:-0 1})
HARD_CUTOFFS=(${HARD_CUTOFFS:-7.0})
INTERACTION_LAYERS=(${INTERACTION_LAYERS:-1})
DEVICES=(${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7})
PARALLEL_CONFIGS="${PARALLEL_CONFIGS:-${#DEVICES[@]}}"
RING_GRAPH_CONFIGS=(${RING_GRAPH_CONFIGS:- \
    2d:1:1 2d:1:2 2d:1:3 2d:1:4 \
    2d:2:1 2d:2:2 2d:2:3 2d:2:4 \
    2d:3:1 2d:3:2 2d:3:3 2d:3:4 \
    2d:4:1 2d:4:2 2d:4:3 2d:4:4 \
})

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

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
nvidia-smi -L

"$PYTHON_BIN" benchmarks/run_models/sweep.py \
    --dataset rotating_ring \
    --ring-n-graphs "$RING_N_GRAPHS" \
    --ring-seed "$RING_SEED" \
    --ring-inner-radius "$RING_INNER_RADIUS" \
    --ring-outer-gap "$RING_OUTER_GAP" \
    --ring-graph-configs "${RING_GRAPH_CONFIGS[@]}" \
    --epochs "$EPOCHS" \
    --seeds "${SEEDS[@]}" \
    --hard-cutoffs "${HARD_CUTOFFS[@]}" \
    --interaction-layers "${INTERACTION_LAYERS[@]}" \
    --model-configs \
        l2_n2 l2_n3 l2_n4 l2_n5 \
        l3_n2 l3_n3 l3_n4 l3_n5 \
    --model hiphop \
    --neighborhood-cutoff edges \
    --output-dir "$OUTPUT_DIR" \
    --results-json-dir "$RESULTS_JSON_DIR" \
    --parallel-configs "$PARALLEL_CONFIGS" \
    --devices "${DEVICES[@]}"
