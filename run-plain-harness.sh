#!/bin/bash
set -euo pipefail

REPO_DIR="/home/t-swsingh/proof-model/ours/verified-code-gen/benchmarks/verus-proof-synthesis"
TASKS_DIR="${REPO_DIR}/benchmarks/VeruSAGE-Bench/tasks"
CONFIG="${REPO_DIR}/verusage/vllm_config.json"
HARNESS="${REPO_DIR}/plain-harness.py"
REPAIR_STEPS=5
WORKERS=${WORKERS:-3}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

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
    BASE_DIR="/home/t-swsingh/proof-model/ours/verified-code-gen/evals/VeruSageBench/plain-harness-qwen3-30B-A3B-instruct-full"
    OUTPUT_DIR="${BASE_DIR}-${TIMESTAMP}"
    LOG_FILE="${OUTPUT_DIR}/benchmark-${TIMESTAMP}.log"
    mkdir -p "${OUTPUT_DIR}"
    EXTRA_ARGS="--output-dir ${OUTPUT_DIR}"
    MODE="fresh"
fi

echo "=== Plain Harness Benchmark Run (${MODE}) ===" | tee "${LOG_FILE}"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "Tasks dir: ${TASKS_DIR}" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Output dir: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Repairs: ${REPAIR_STEPS}" | tee -a "${LOG_FILE}"
echo "Workers: ${WORKERS}" | tee -a "${LOG_FILE}"
echo "================================================" | tee -a "${LOG_FILE}"

# shellcheck disable=SC2086
PYTHONUNBUFFERED=1 python3 "${HARNESS}" \
  --config "${CONFIG}" \
  --tasks-dir "${TASKS_DIR}" \
  ${EXTRA_ARGS} \
  --repairs "${REPAIR_STEPS}" \
  --workers "${WORKERS}" \
  2>&1 | tee -a "${LOG_FILE}"