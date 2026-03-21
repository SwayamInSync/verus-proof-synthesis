#!/bin/bash
set -euo pipefail

TASKS_DIR="../benchmarks/VeruSAGE-Bench/tasks"
CONFIG="vllm_config.json"
REPAIR_STEPS=5
WORKERS=${WORKERS:-3}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Usage: ./run_bechmark.sh                          # fresh run
#        ./run_bechmark.sh /path/to/previous/results # continue from previous run

CONTINUE_FROM="${1:-}"

if [[ -n "${CONTINUE_FROM}" ]]; then
    if [[ ! -d "${CONTINUE_FROM}" ]]; then
        echo "Error: Directory not found: ${CONTINUE_FROM}"
        exit 1
    fi
    OUTPUT_DIR="${CONTINUE_FROM}"
    LOG_FILE="${OUTPUT_DIR}/benchmark-continue-${TIMESTAMP}.log"
    EXTRA_ARGS="--continue-from ${CONTINUE_FROM}"
    MODE="continue"
else
    BASE_DIR="/home/t-swsingh/proof-model/ours/verified-code-gen/evals/VeruSageBench/results-qwen3b-full"
    OUTPUT_DIR="${BASE_DIR}-${TIMESTAMP}"
    LOG_FILE="${OUTPUT_DIR}/benchmark-${TIMESTAMP}.log"
    mkdir -p "${OUTPUT_DIR}"
    EXTRA_ARGS="--output-dir ${OUTPUT_DIR}"
    MODE="fresh"
fi

echo "=== VeruSAGE Benchmark Run (${MODE}) ===" | tee "${LOG_FILE}"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "Tasks dir: ${TASKS_DIR}" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Output dir: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Repairs: ${REPAIR_STEPS}" | tee -a "${LOG_FILE}"
echo "Workers: ${WORKERS}" | tee -a "${LOG_FILE}"
echo "===========================================" | tee -a "${LOG_FILE}"

# shellcheck disable=SC2086
PYTHONUNBUFFERED=1 python3 run_batch_concurrent.py \
  "${TASKS_DIR}" \
  "${CONFIG}" \
  ${EXTRA_ARGS} \
  --repair "${REPAIR_STEPS}" \
  --workers "${WORKERS}" \
  2>&1 | tee -a "${LOG_FILE}"