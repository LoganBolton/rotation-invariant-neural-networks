#!/usr/bin/env bash
#SBATCH --job-name=ring-pair-no-gn
#SBATCH --output=benchmarks/rotating_ring/results/slurm_logs/ring_pair_no_group_norm_%j.out
#SBATCH --error=benchmarks/rotating_ring/results/slurm_logs/ring_pair_no_group_norm_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=volta-x86
#SBATCH --constraint=gpu_count:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/vast/home/logan_bolton/Github/rotation-invariant-neural-networks}"
cd "$REPO_ROOT"

export HIPHOP_SOURCE_ROOT="${HIPHOP_SOURCE_ROOT:-/vast/home/logan_bolton/Github/hippynn-basic-lmax3-nmax4}"
export HIPHOP_INVARIANTS_REF="${HIPHOP_INVARIANTS_REF:-lmax3-basis-invariants-tests}"
export HIPHOP_GROUP_NORM=false
export RING_Z_PHASE_SAMPLE=false
export OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/rotating_ring/results/two_graph_radius1_no_gap_basic_l2_l3_n4_all_sizes_no_group_norm}"
export RESULTS_JSON_DIR="${RESULTS_JSON_DIR:-$OUTPUT_DIR/json_results}"

exec "$REPO_ROOT/benchmarks/rotating_ring/run_z_rotation_basic_l2_l3_n4_all_sizes.sh"
