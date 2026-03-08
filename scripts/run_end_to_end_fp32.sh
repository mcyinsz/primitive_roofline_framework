#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

THREADS=36
SEED=42
TAG=""
SHOW_HW_IN_PLOT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --show-hw-in-plot)
            SHOW_HW_IN_PLOT=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--threads N] [--seed N] [--tag run_tag] [--show-hw-in-plot]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${TAG}" ]]; then
    TAG="$(date +%Y%m%d_%H%M%S)"
fi

ROOFLINE_JSON="results/processed/roofline_fp32_${TAG}.json"
RAW_CSV="results/raw/suite_fp32_raw_${TAG}.csv"
HW_SIM_CSV="results/raw/suite_fp32_hw_sim_${TAG}.csv"
SUMMARY_CSV="results/processed/suite_fp32_summary_${TAG}.csv"
PLOT_PNG="results/plots/roofline_fp32_primitives_${TAG}.svg"

mkdir -p results/raw results/processed results/plots

echo "[1/6] build benchmark"
make bench

echo "[2/6] compute roofline constants"
python3 scripts/compute_roofline.py \
    --profile profiles/current_server.json \
    --precision fp32 \
    --output "${ROOFLINE_JSON}"

echo "[3/6] run primitive suite (model metrics)"
python3 scripts/run_suite.py \
    --config configs/suite_fp32.csv \
    --output "${RAW_CSV}" \
    --bench bin/primitive_bench \
    --threads "${THREADS}" \
    --seed "${SEED}"

echo "[4/6] run primitive suite (hardware simulation metrics)"
python3 scripts/run_suite_hw_sim.py \
    --config configs/suite_fp32.csv \
    --output "${HW_SIM_CSV}" \
    --bench bin/primitive_bench \
    --threads "${THREADS}" \
    --seed "${SEED}"

echo "[5/6] summarize suite"
python3 scripts/summarize_suite.py \
    --roofline "${ROOFLINE_JSON}" \
    --raw "${RAW_CSV}" \
    --hw-sim "${HW_SIM_CSV}" \
    --output "${SUMMARY_CSV}"

echo "[6/6] plot roofline + primitive points"
PLOT_ARGS=(
    --roofline "${ROOFLINE_JSON}"
    --summary "${SUMMARY_CSV}"
    --output "${PLOT_PNG}"
    --title "FP32 Roofline + Primitive Points (${TAG})"
)
if [[ "${SHOW_HW_IN_PLOT}" -eq 1 ]]; then
    PLOT_ARGS+=(--show-hw)
fi
python3 scripts/plot_roofline.py "${PLOT_ARGS[@]}"

echo
echo "Run complete."
echo "roofline : ${ROOFLINE_JSON}"
echo "raw      : ${RAW_CSV}"
echo "hw-sim   : ${HW_SIM_CSV}"
echo "summary  : ${SUMMARY_CSV}"
echo "plot     : ${PLOT_PNG}"
