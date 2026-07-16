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
MAX_CONCURRENT="${MAX_CONCURRENT:-${#DEVICES[@]}}"
RING_GRAPH_CONFIGS=(${RING_GRAPH_CONFIGS:- \
    2d_1inner_1_outer 2d_1inner_2_outer 2d_1inner_3_outer 2d_1inner_4_outer \
    2d_2inner_1_outer 2d_2inner_2_outer 2d_2inner_3_outer 2d_2inner_4_outer \
    2d_3inner_1_outer 2d_3inner_2_outer 2d_3inner_3_outer 2d_3inner_4_outer \
    2d_4inner_1_outer 2d_4inner_2_outer 2d_4inner_3_outer 2d_4inner_4_outer \
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

active_jobs=0
job_index=0

wait_for_slot() {
    while (( active_jobs >= MAX_CONCURRENT )); do
        wait -n
        active_jobs=$((active_jobs - 1))
    done
}

run_graph_config() {
    local graph_config="$1"
    local device="$2"
    local visible_device=""
    local train_device="$device"

    if [[ "$device" == cuda:* ]]; then
        visible_device="${device#cuda:}"
        train_device="cuda:0"
    fi

    echo "[run] ${graph_config} on ${device}"
    env ${visible_device:+CUDA_VISIBLE_DEVICES="$visible_device"} \
        "$PYTHON_BIN" benchmarks/run_models/sweep.py \
        --dataset rotating_ring \
        --ring-n-graphs "$RING_N_GRAPHS" \
        --ring-seed "$RING_SEED" \
        --ring-inner-radius "$RING_INNER_RADIUS" \
        --ring-outer-gap "$RING_OUTER_GAP" \
        --ring-graph-configs "$graph_config" \
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
        --device "$train_device"
}

for graph_config in "${RING_GRAPH_CONFIGS[@]}"; do
    wait_for_slot
    device="${DEVICES[$((job_index % ${#DEVICES[@]}))]}"
    run_graph_config "$graph_config" "$device" &
    active_jobs=$((active_jobs + 1))
    job_index=$((job_index + 1))
done

wait
